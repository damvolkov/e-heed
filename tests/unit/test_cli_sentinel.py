"""Tests for eheed sentinel CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from e_heed.shared.settings import AppConfig

##### AVAILABLE MODELS #####


def test_available_models_discovers_onnx(tmp_path: Path) -> None:
    ww_dir = tmp_path / "ww"
    ww_dir.mkdir()
    (ww_dir / "alexa.onnx").touch()
    (ww_dir / "eager.onnx").touch()
    (ww_dir / "config.yaml").touch()

    with patch("e_heed.cli.commands.sentinel._MODELS_DIR", ww_dir):
        from e_heed.cli.commands.sentinel import _available_models

        models = _available_models()

    assert models == ["alexa", "eager"]


def test_available_models_empty_when_no_dir(tmp_path: Path) -> None:
    with patch("e_heed.cli.commands.sentinel._MODELS_DIR", tmp_path / "nonexistent"):
        from e_heed.cli.commands.sentinel import _available_models

        assert _available_models() == []


##### LIST MODELS #####


def test_sentinel_list_models(tmp_path: Path) -> None:
    ww_dir = tmp_path / "ww"
    ww_dir.mkdir()
    (ww_dir / "alexa.onnx").touch()
    (ww_dir / "maia.onnx").touch()

    with patch("e_heed.cli.commands.sentinel._MODELS_DIR", ww_dir):
        from e_heed.cli.commands.sentinel import sentinel

        sentinel(list_models=True)


##### MODEL OVERRIDE #####


@patch("e_heed.cli.commands.sentinel.asyncio.run")
@patch("e_heed.cli.commands.sentinel._available_models", return_value=["alexa", "eager", "maia"])
@patch("e_heed.cli.commands.sentinel.AppConfig.load")
async def test_sentinel_model_override(
    mock_load: MagicMock, mock_models: MagicMock, mock_asyncio_run: MagicMock
) -> None:
    config = AppConfig()
    mock_load.return_value = config

    from e_heed.cli.commands.sentinel import sentinel

    sentinel(model="eager")

    assert config.sentinel.wakeword.model == "eager"
    mock_asyncio_run.assert_called_once()


@patch("e_heed.cli.commands.sentinel._available_models", return_value=["alexa", "eager"])
@patch("e_heed.cli.commands.sentinel.AppConfig.load")
async def test_sentinel_invalid_model_exits(mock_load: MagicMock, mock_models: MagicMock) -> None:
    mock_load.return_value = AppConfig()

    from e_heed.cli.commands.sentinel import sentinel

    with pytest.raises(SystemExit) as exc_info:
        sentinel(model="nonexistent")
    assert exc_info.value.code == 1


##### THRESHOLD OVERRIDE #####


@patch("e_heed.cli.commands.sentinel.asyncio.run")
@patch("e_heed.cli.commands.sentinel.AppConfig.load")
async def test_sentinel_threshold_override(mock_load: MagicMock, mock_asyncio_run: MagicMock) -> None:
    config = AppConfig()
    mock_load.return_value = config

    from e_heed.cli.commands.sentinel import sentinel

    sentinel(threshold=0.3)

    assert config.sentinel.wakeword.threshold == 0.3


##### SENTINEL CLI ENTRY #####


@patch("e_heed.cli.commands.sentinel.AppConfig.load")
async def test_sentinel_invalid_config_exits(mock_load: MagicMock) -> None:
    from pydantic import ValidationError

    mock_load.side_effect = ValidationError.from_exception_data("AppConfig", [])

    from e_heed.cli.commands.sentinel import sentinel

    with pytest.raises(SystemExit) as exc_info:
        sentinel()
    assert exc_info.value.code == 1


@patch("e_heed.cli.commands.sentinel.asyncio.run")
@patch("e_heed.cli.commands.sentinel.AppConfig.load")
async def test_sentinel_calls_asyncio_run(mock_load: MagicMock, mock_asyncio_run: MagicMock) -> None:
    mock_load.return_value = AppConfig()

    from e_heed.cli.commands.sentinel import sentinel

    sentinel()

    mock_asyncio_run.assert_called_once()


##### SENTINEL LOOP LOGIC #####


async def test_sentinel_loop_starts_audio_and_runs() -> None:
    """Verify _run_sentinel_loop starts audio, enters sentinel, and can be interrupted."""
    from e_heed.cli.commands.sentinel import _run_sentinel_loop

    config = AppConfig()

    with (
        patch("e_heed.shared.logger.configure_logging"),
        patch("e_heed.daemon.adapters.audio.AudioAdapter") as mock_audio_cls,
        patch("e_heed.daemon.sentinel.pipeline.SentinelPipeline") as mock_sentinel_cls,
    ):
        mock_audio = MagicMock()
        mock_audio.start = AsyncMock()
        mock_audio.stop = AsyncMock()
        mock_audio.queue = MagicMock()
        mock_audio_cls.return_value = mock_audio

        mock_sentinel = MagicMock()
        mock_sentinel.wakeword_name = "alexa"
        mock_sentinel.last_event = None
        mock_sentinel.stop = AsyncMock()
        mock_sentinel.run = AsyncMock(side_effect=KeyboardInterrupt)
        mock_sentinel_cls.return_value = mock_sentinel

        with pytest.raises(KeyboardInterrupt):
            await _run_sentinel_loop(config)

        mock_audio.start.assert_awaited_once()
        mock_sentinel.run.assert_awaited_once()
