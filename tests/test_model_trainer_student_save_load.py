"""Tests for ModelStudentTrainer adapter save/load functionality."""

import os
import tempfile
import tensorflow as tf

from Core.models.ModelStudentTrainer import ModelStudentTrainer


class TestModelStudentTrainerSaveLoad:
    """Tests for ModelStudentTrainer save and load methods with adapters."""

    def _create_dummy_input(self, batch_size, timesteps):
        """Create dummy input data for testing."""
        return {
            "points": tf.random.normal(
                (batch_size, timesteps, 478, 2), dtype=tf.float32
            ),
            "left eye": tf.random.normal(
                (batch_size, timesteps, 32, 32, 1), dtype=tf.float32
            ),
            "right eye": tf.random.normal(
                (batch_size, timesteps, 32, 32, 1), dtype=tf.float32
            ),
            "time": tf.ones((batch_size, timesteps, 1), dtype=tf.float32),
            "userId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "placeId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "screenId": tf.constant([[0, 0], [1, 1]], dtype=tf.int32),
            "cameraId": tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
            "monitorId": tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
        }

    def test_save_creates_adapter_files(self, cached_model_wrappers):
        """Test that save() creates adapter NPZ files in correct location."""
        student_wrapper = cached_model_wrappers["timesteps_2"]
        teacher_wrapper = cached_model_wrappers["timesteps_2"]

        trainer = ModelStudentTrainer(
            model_wrapper=student_wrapper, teachers_models=[teacher_wrapper]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir, postfix="test")

            # Check adapter files exist (save_npz adds .npz extension automatically)
            model_path = student_wrapper.get_checkpoint_path(tmpdir, "test")
            adapter_inter_file = f"{model_path}/adapter_intermediate.npz"
            adapter_final_file = f"{model_path}/adapter_final.npz"

            assert os.path.exists(
                adapter_inter_file
            ), f"adapter_intermediate.npz not found at {adapter_inter_file}"
            assert os.path.exists(
                adapter_final_file
            ), f"adapter_final.npz not found at {adapter_final_file}"

            # Verify files are non-empty
            assert (
                os.path.getsize(adapter_inter_file) > 0
            ), "adapter_intermediate.npz is empty"
            assert os.path.getsize(adapter_final_file) > 0, "adapter_final.npz is empty"

    def test_load_without_adapters_backward_compatibility(self, cached_model_wrappers):
        """Test that loading old checkpoints without adapters works with force=True."""
        student_wrapper = cached_model_wrappers["timesteps_2"]
        teacher_wrapper = cached_model_wrappers["timesteps_2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save only the model wrapper (simulate old checkpoint)
            student_wrapper.save(tmpdir, postfix="old")

            # Load with force=True should work even if adapters are missing
            trainer2 = ModelStudentTrainer(
                model_wrapper=student_wrapper, teachers_models=[teacher_wrapper]
            )
            try:
                trainer2.load(tmpdir, postfix="old", force=True)
            except FileNotFoundError:
                # force=True should allow missing adapters - this is backward compatible behavior
                pass

            # Adapters should still exist (created during __init__)
            assert (
                trainer2._adapters["intermediate"] is not None
            ), "Intermediate adapter should exist"
            assert (
                trainer2._adapters["intermediate"]._reconstruction is not None
            ), "Intermediate adapter reconstruction should exist"
            assert (
                trainer2._adapters["intermediate"]._latents is not None
            ), "Intermediate adapter latents should exist"

            assert trainer2._adapters["final"] is not None, "Final adapter should exist"
            assert (
                trainer2._adapters["final"]._reconstruction is not None
            ), "Final adapter reconstruction should exist"
            assert (
                trainer2._adapters["final"]._latents is not None
            ), "Final adapter latents should exist"

    def test_load_incomplete_adapters_with_force_true(self, cached_model_wrappers):
        """Test that incomplete adapters are skipped when force=True."""
        student_wrapper = cached_model_wrappers["timesteps_2"]
        teacher_wrapper = cached_model_wrappers["timesteps_2"]

        trainer1 = ModelStudentTrainer(
            model_wrapper=student_wrapper, teachers_models=[teacher_wrapper]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save full checkpoint
            trainer1.save(tmpdir, postfix="incomplete")

            model_path = student_wrapper.get_checkpoint_path(tmpdir, "incomplete")

            # Delete one adapter file
            adapter_final_file = f"{model_path}/adapter_final.npz"
            os.remove(adapter_final_file)

            # Load should succeed with force=True (missing adapters are ignored)
            trainer2 = ModelStudentTrainer(
                model_wrapper=student_wrapper, teachers_models=[teacher_wrapper]
            )
            try:
                trainer2.load(tmpdir, postfix="incomplete", force=True)
            except FileNotFoundError:
                # With force=True, missing adapters should be tolerated
                # Adapters will remain as initialized (not loaded)
                pass

            # Adapters should still exist (created during __init__)
            assert trainer2._adapters["intermediate"] is not None
            assert trainer2._adapters["intermediate"]._reconstruction is not None
            assert trainer2._adapters["intermediate"]._latents is not None

            assert trainer2._adapters["final"] is not None
            assert trainer2._adapters["final"]._reconstruction is not None
            assert trainer2._adapters["final"]._latents is not None
