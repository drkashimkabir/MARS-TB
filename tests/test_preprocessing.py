import cv2
import numpy as np
import pytest
import torch

from mars_tb.data.preprocessing import (
    IMAGENET_MEAN,
    ImageRejected,
    PreprocessConfig,
    apply_clahe,
    load_and_preprocess,
    normalize,
    preprocess_array,
    quality_check,
    read_image,
    to_uint8,
)


def _write(tmp_path, arr, name="x.png"):
    p = tmp_path / name
    cv2.imwrite(str(p), arr)
    return p


def test_output_is_512_three_channel_float_in_unit_range(tmp_path):
    p = _write(tmp_path, (np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8))
    x = load_and_preprocess(p)
    assert x.shape == (3, 512, 512) and x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])  # channel replication


def test_resize_does_not_pad_or_preserve_aspect(tmp_path):
    img = np.full((256, 512), 128, np.uint8)
    x = load_and_preprocess(_write(tmp_path, img))
    assert x.shape[-2:] == (512, 512)


def test_8bit_and_16bit_scaling(tmp_path):
    u8 = np.full((64, 64), 255, np.uint8)
    u16 = np.full((64, 64), 65535, np.uint16)
    assert read_image(_write(tmp_path, u8, "a.png")).max() == pytest.approx(1.0)
    assert read_image(_write(tmp_path, u16, "b.png")).max() == pytest.approx(1.0)
    half16 = np.full((64, 64), 32768, np.uint16)
    assert read_image(_write(tmp_path, half16, "c.png")).mean() == pytest.approx(32768 / 65535, abs=1e-6)


def test_clahe_runs_on_uint8_with_manuscript_parameters_and_raises_contrast():
    rng = np.random.default_rng(0)
    low = (110 + rng.integers(0, 20, (256, 256))).astype(np.uint8)
    out = apply_clahe(low, PreprocessConfig())
    assert out.dtype == np.uint8 and out.std() > low.std()
    with pytest.raises(ValueError):
        apply_clahe(low.astype(np.float32))


def test_clahe_matches_opencv_reference_call():
    rng = np.random.default_rng(1)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    ref = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    assert np.array_equal(apply_clahe(img), ref)


@pytest.mark.parametrize("h,w", [(255, 600), (600, 255), (200, 200)])
def test_quality_filter_rejects_small_images(h, w):
    with pytest.raises(ImageRejected):
        quality_check(h, w)


@pytest.mark.parametrize("h,w", [(300, 700), (700, 300)])
def test_quality_filter_rejects_extreme_aspect_ratio(h, w):
    with pytest.raises(ImageRejected, match="aspect"):
        quality_check(h, w)


def test_quality_filter_accepts_boundary_values():
    quality_check(256, 512)  # shorter edge == 256, aspect == 2.0
    quality_check(512, 256)  # aspect == 0.5


def test_normalisation_uses_imagenet_statistics():
    x = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).expand(3, 4, 4).clone()
    assert torch.allclose(normalize(x), torch.zeros(3, 4, 4), atol=1e-6)
    ones = torch.ones(3, 2, 2)
    assert normalize(ones)[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229)


def test_unreadable_and_missing_files(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        read_image(tmp_path / "bad.png")
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")


def test_to_uint8_rounds_and_clips():
    assert to_uint8(np.array([-0.5, 0.0, 0.5, 1.0, 2.0])).tolist() == [0, 0, 128, 255, 255]


def test_preprocess_array_requires_2d():
    with pytest.raises(ValueError):
        preprocess_array(np.zeros((300, 300, 3), np.float32))


def _make_dicom(path, arr, photometric, bits):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Synthetic^Test"  # must never leak through read_image
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, bits, bits - 1, 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True) if hasattr(pydicom, "__version__") and int(
        pydicom.__version__.split(".")[0]
    ) >= 3 else ds.save_as(str(path))
    return path


def test_dicom_12bit_is_scaled_by_bits_stored_not_65535(tmp_path):
    arr = np.full((300, 300), 2048, np.uint16)  # mid-grey for a 12-bit detector
    got = read_image(_make_dicom(tmp_path / "a.dcm", arr, "MONOCHROME2", 12))
    assert got.mean() == pytest.approx(2048 / 4095, abs=1e-6)


def test_dicom_monochrome1_is_inverted(tmp_path):
    arr = np.zeros((300, 300), np.uint16)
    arr[:, :150] = 4095
    got = read_image(_make_dicom(tmp_path / "b.dcm", arr, "MONOCHROME1", 12))
    assert got[:, :150].mean() == pytest.approx(0.0) and got[:, 150:].mean() == pytest.approx(1.0)
