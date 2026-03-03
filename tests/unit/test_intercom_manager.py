"""Unit tests for IntercomProcess — arg building, pear discovery, lifecycle."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from transport.intercom_manager import IntercomProcess, _find_pear


class TestPearDiscovery:
    def test_configured_path_takes_priority(self):
        result = _find_pear("/custom/bin/pear")
        assert result == "/custom/bin/pear"

    @patch("shutil.which", return_value="/usr/local/bin/pear")
    def test_which_fallback(self, mock_which):
        result = _find_pear("")
        assert result == "/usr/local/bin/pear"

    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    @patch("os.access", return_value=False)
    def test_bare_fallback(self, mock_access, mock_isfile, mock_which):
        result = _find_pear("")
        assert result == "pear"


class TestBuildArgs:
    def test_default_args(self):
        proc = IntercomProcess(
            channel="test-channel",
            sc_bridge_port=49222,
            sc_bridge_token="abc123",
            inviter_keys="deadbeef",
            pear_path="/usr/local/bin/pear",
            intercom_dir="/tmp/intercom",
        )
        args = proc._build_args()

        assert args[0] == "/usr/local/bin/pear"
        assert args[1] == "run"
        assert args[2] == "/tmp/intercom"
        assert "--peer-store-name" in args
        assert args[args.index("--peer-store-name") + 1] == "gateway"
        assert "--sc-bridge" in args
        assert args[args.index("--sc-bridge") + 1] == "1"
        assert "--sc-bridge-host" in args
        assert args[args.index("--sc-bridge-host") + 1] == "127.0.0.1"
        assert "--sc-bridge-port" in args
        assert args[args.index("--sc-bridge-port") + 1] == "49222"
        assert "--sc-bridge-token" in args
        assert args[args.index("--sc-bridge-token") + 1] == "abc123"
        assert "--sidechannels" in args
        assert args[args.index("--sidechannels") + 1] == "test-channel"
        assert "--sidechannel-inviter-keys" in args
        assert args[args.index("--sidechannel-inviter-keys") + 1] == "deadbeef"
        assert "--sidechannel-quiet" in args

    def test_custom_store_name(self):
        proc = IntercomProcess(
            channel="ch",
            sc_bridge_port=1234,
            sc_bridge_token="tok",
            inviter_keys="key",
            pear_path="/bin/pear",
            intercom_dir="/tmp/ic",
            peer_store_name="robot",
        )
        args = proc._build_args()
        assert args[args.index("--peer-store-name") + 1] == "robot"


class TestLifecycle:
    def test_initial_state(self):
        proc = IntercomProcess(
            channel="ch", sc_bridge_port=1, sc_bridge_token="t",
            inviter_keys="k", pear_path="/bin/pear",
        )
        assert not proc.is_running
        assert proc._process is None
        assert not proc._stopping
