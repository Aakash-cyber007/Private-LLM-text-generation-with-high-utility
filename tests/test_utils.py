import pytest
import numpy as np
import pandas as pd
import math
from unittest.mock import MagicMock, patch

import invisibleink.utils as utils


"""Test preprocess() for whitespace removal, type coercion, and handling of None/NaN input values"""
@pytest.mark.parametrize("input_val, expected", [ 
    ("  \nHello\t  World  ", "Hello World"),      # Newlines, tabs, and multi-spaces
    (None, ""), (pd.NA, ""), ("", ""),            # None, NaN and empty string check
    (123, "123"), (3.14, "3.14"),                 # Integer and Float coercion
    ([1, 2, 3], "[1, 2, 3]")                      # List coercion
])
def test_preprocess(input_val, expected):
    assert utils.preprocess(input_val) == expected


"""Verifies non-string/empty path raises ValueError."""
@pytest.mark.parametrize("invalid_path", [123, "", "   "])
def test_load_hf_tokenizer_invalid_path(invalid_path):
    with pytest.raises(ValueError) as excinfo:
        utils.load_hf_tokenizer(invalid_path)
    assert "name_or_path must be a non-empty string or path." in str(excinfo.value)


"""Verifies FileNotFoundError when model is not local and download is disabled."""
def test_load_hf_tokenizer_download_disabled(mocker):
    mocker.patch('os.path.isdir', return_value=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        utils.load_hf_tokenizer("test/model", allow_download=False)
    assert "not found locally and downloads disabled" in str(excinfo.value)


"""Verifies non-string/empty path raises ValueError."""
@pytest.mark.parametrize("invalid_path", [123, "", "   "])
def test_load_hf_model_invalid_path(invalid_path):
    with pytest.raises(ValueError) as excinfo:
        utils.load_hf_model(invalid_path)
    assert "name_or_path must be a non-empty string or path." in str(excinfo.value)


"""Verifies FileNotFoundError when model is not local and download is disabled."""
def test_load_hf_model_download_disabled(mocker):
    mocker.patch('os.path.isdir', return_value=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        utils.load_hf_model("test/model", allow_download=False)
    assert "not found locally and downloads disabled" in str(excinfo.value)


"""Verifies tokenizer without apply_chat_template raises AttributeError."""
def test_get_prompt_tokenizer_nochat():
    mock_tokenizer = MagicMock()
    del mock_tokenizer.apply_chat_template
    with pytest.raises(AttributeError) as excinfo:
        utils.get_prompt(mock_tokenizer, "desc")
    assert "Tokenizer does not support apply_chat_template()." in str(excinfo.value)


"""Verifies list too small for requested batches raises ValueError."""
def test_batchify_list_too_small():
    test_list = [1, 2, 3, 4]
    with pytest.raises(ValueError) as excinfo:
        list(utils.batchify(test_list, s=3, n=2))
    assert "List too small for creating 2 batches of size 3 (len=4)" in str(excinfo.value)


"""Verifies correct batch slicing and return types (list)."""
def test_batchify():
    test_list = list(range(10))
    batches = list(utils.batchify(test_list, s=3, n=3))
    
    assert len(batches) == 3
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert isinstance(batches[0], list)


"""Verifies correct aggregation when lens, stds are None."""
def test_combined_mean_std_uniform():
    mean, std = utils.combined_mean_std(means=[10, 20], stds=None, lens=None)
    assert mean == 15.0
    assert std == 5.0


"""Verifies correct aggregation with non-zero stds and different lens."""
def test_combined_mean_std_weighted():
    means = [10.0, 20.0]
    stds = [1.0, 2.0]
    lens = [10, 30]
    mean, std = utils.combined_mean_std(means, stds, lens)
    
    assert math.isclose(mean, 17.5)
    assert math.isclose(std, math.sqrt(22.0))


"""Verifies difference_clipping logic."""
def test_difference_clip():
    logit = np.array([[5.0, 0.0, -5.0], [-5.0, 0.0, 5.0]])
    publogit = np.array([1.0, 0.0, -1.0])
    clip_norm = 2.0
    
    expected = np.array([[3.0, 0.0, -3.0], [-1.0, 0.0, 1.0]])
    result = utils.difference_clip(logit, publogit, clip_norm)
    assert np.allclose(result, expected)


"""Verifies k > size of array raises ValueError."""
def test_get_topk_large():
    with pytest.raises(ValueError) as excinfo:
        utils.get_topk(np.array([1, 2, 3]), 4, 0.1, 1)
    assert "k cannot be larger than number of elements in pub." in str(excinfo.value)


"""Verifies top-k+ sampling"""
def test_get_topk():
    pub_logits = np.array([0, 1, 2, 3, 4])
    k, clip, batch = 2, 1.0, 2
    
    mask_pub, idxs = utils.get_topk(pub_logits, k, clip, batch)
    expected_mask = np.array([False, False, True, True, True])
    expected_idxs = np.array([2, 3])
    
    assert np.array_equal(mask_pub, expected_mask)
    assert np.array_equal(idxs, expected_idxs)


"""Verifies rho=0 correctly returns delta=0."""
def test_cdp_delta_zero_rho():
    assert utils.cdp_delta(0.0, 10.0) == 0.0

"""Verifies rho=0 correctly returns eps=0."""
def test_cdp_eps_zero_rho():
    assert utils.cdp_eps(0.0, 1e-6) == 0.0

"""Verifies eps=0 correctly returns rho=0."""
def test_cdp_rho_zero_eps():
    assert utils.cdp_rho(0.0, 1e-6) == 0.0


"""Verifies the rho-zCDP accounting."""
def test_compute_rho():
    num_toks = 100
    clip_norm = 0.5
    batch_size = 2
    temp = 1.0
    
    expected_rho = 3.125
    result = utils.compute_rho(num_toks, clip_norm, batch_size, temp)
    assert math.isclose(result, expected_rho)


"""Verifies correctness of get_epsilon calls."""
def test_get_epsilon_calls(mocker):
    mock_compute_rho = mocker.patch('invisibleink.utils.compute_rho', return_value=1.0)
    mock_cdp_eps = mocker.patch('invisibleink.utils.cdp_eps', return_value=5.0)
    
    utils.get_epsilon(num_toks=10, clip_norm=1.0, batch_size=5, temp=1.0, delta=1e-5)
    
    # batch_size should be passed as B-1 = 4
    mock_compute_rho.assert_called_with(10, 1.0, 4, 1.0)
    mock_cdp_eps.assert_called_with(1.0, 1e-5)


"""Verifies get_clip calls cdp_rho."""
def test_get_clip_calls_cdp_rho(mocker):
    mock_cdp_rho = mocker.patch('invisibleink.utils.cdp_rho', return_value=2.0)
    
    # Calculate expected rho_tok: 2.0 / 10 = 0.2
    # Calculate expected clip: 1.0 * (5-1) * sqrt(2 * 0.2) = 4.0 * sqrt(0.4)
    expected_clip = 4.0 * math.sqrt(0.4)
    result = utils.get_clip(epsilon=1.0, num_toks=10, temp=1.0, batch_size=5, delta=1e-6)
    
    mock_cdp_rho.assert_called_with(1.0, 1e-6)
    assert math.isclose(result, expected_clip)