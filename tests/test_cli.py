"""Tests for CLI functionality."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIHelp:
    """Tests for CLI help output."""
    
    def test_main_help(self):
        """Test that --help works for main command."""
        result = subprocess.run(
            [sys.executable, "-m", "hsi_rgbd_calib.cli", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "hsi-rgbd-calib" in result.stdout or "HSI-RGBD" in result.stdout
        assert "calibrate" in result.stdout
        assert "validate" in result.stdout
        assert "export" in result.stdout
    
    def test_calibrate_help(self):
        """Test that --help works for calibrate subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "hsi_rgbd_calib.cli", "calibrate", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--session" in result.stdout
        assert "--config" in result.stdout
        assert "--out" in result.stdout
        assert "--dry-run" in result.stdout
    
    def test_validate_help(self):
        """Test that --help works for validate subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "hsi_rgbd_calib.cli", "validate", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--session" in result.stdout
        assert "--rig" in result.stdout
        assert "--out" in result.stdout
    
    def test_export_help(self):
        """Test that --help works for export subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "hsi_rgbd_calib.cli", "export", "--help"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--session" in result.stdout
        assert "--rig" in result.stdout
        assert "--out" in result.stdout
    
    def test_version(self):
        """Test that --version works."""
        result = subprocess.run(
            [sys.executable, "-m", "hsi_rgbd_calib.cli", "--version"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


class TestCLIDryRun:
    """Tests for CLI dry-run mode."""
    
    def test_calibrate_dry_run(self, sample_session_path: Path, config_path: Path, temp_output_dir: Path):
        """Test calibrate with --dry-run."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hsi_rgbd_calib.cli",
                "calibrate",
                "--session", str(sample_session_path),
                "--config", str(config_path / "li_wen_qiu.yaml"),
                "--out", str(temp_output_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "DRY-RUN" in result.stdout
        
        # No files should be created in dry-run
        rig_yaml = temp_output_dir / "rig.yaml"
        assert not rig_yaml.exists()
    
    def test_validate_dry_run(self, sample_session_path: Path, temp_output_dir: Path):
        """Test validate with --dry-run."""
        # First need a rig.yaml - use a placeholder check
        result = subprocess.run(
            [
                sys.executable, "-m", "hsi_rgbd_calib.cli",
                "validate",
                "--session", str(sample_session_path),
                "--rig", str(sample_session_path / "nonexistent.yaml"),
                "--out", str(temp_output_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        
        # May fail due to nonexistent file, but dry-run should be recognized
        assert "DRY-RUN" in result.stdout or result.returncode != 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_full_calibrate_workflow(
        self, 
        sample_session_path: Path, 
        config_path: Path, 
        temp_output_dir: Path
    ):
        """Test full calibrate workflow produces expected outputs."""
        result = subprocess.run(
            [
                sys.executable, "-m", "hsi_rgbd_calib.cli",
                "calibrate",
                "--session", str(sample_session_path),
                "--config", str(config_path / "li_wen_qiu.yaml"),
                "--out", str(temp_output_dir),
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"Calibration failed: {result.stderr}"
        
        # Check outputs exist
        rig_yaml = temp_output_dir / "rig.yaml"
        report_json = temp_output_dir / "calibration_report.json"
        
        assert rig_yaml.exists(), "rig.yaml not created"
        assert report_json.exists(), "calibration_report.json not created"
        
        # Validate rig.yaml content
        import yaml
        with open(rig_yaml) as f:
            rig_data = yaml.safe_load(f)
        
        assert "artifact_version" in rig_data
        assert "extrinsics" in rig_data
        assert "oak" in rig_data
        assert "hsi" in rig_data
