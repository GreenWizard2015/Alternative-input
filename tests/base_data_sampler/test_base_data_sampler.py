"""Tests for BaseDataSampler frame selection strategies."""


class TestBaseDataSamplerFrameSelection:
    """Tests for BaseDataSampler frame selection strategies."""

    def test_uniform_time_strategy(self, sampler_with_frames):
        """Test uniform time sampling spans time range."""
        result = sampler_with_frames.framesFor(
            mainInd=9, samples=[0, 1, 2, 3], steps=3, stepsSampling="uniform time"
        )

        # Verify frame selection validity: correct length, sorted, no duplicates, contains mainInd
        assert (
            len(result) == 3
        ), f"Expected 3 frames for uniform time sampling, got {len(result)} frames: {result}"
        assert 9 in result, f"mainInd 9 should be included in result, but got {result}"
        assert result == sorted(result), f"Result should be sorted, but got {result}"
        assert len(result) == len(set(result)), f"Result has duplicate frames: {result}"

        # Verify spanning: earliest from samples, latest is mainInd
        assert result[0] in [
            0,
            1,
            2,
            3,
        ], f"First frame should be early, got {result[0]} from {result}"
        assert (
            result[-1] == 9
        ), f"Last frame should be mainInd 9, got {result[-1]} from {result}"

    def test_uniform_strategy(self, sampler_with_frames):
        """Test uniform random sampling from pool."""
        result = sampler_with_frames.framesFor(
            mainInd=7, samples=[0, 1, 2, 3, 5, 6], steps=4, stepsSampling="uniform"
        )

        # Verify frame selection validity
        assert (
            len(result) == 4
        ), f"Expected 4 frames for uniform sampling, got {len(result)} frames: {result}"
        assert 7 in result, f"mainInd 7 should be included in result, but got {result}"
        assert result == sorted(result), f"Result should be sorted, but got {result}"
        assert len(result) == len(set(result)), f"Result has duplicate frames: {result}"

        # Verify all non-mainInd frames come from samples
        sampled = [idx for idx in result if idx != 7]
        assert len(sampled) == 3, f"Should have 3 frames from pool, got {len(sampled)}"
        assert all(
            idx in [0, 1, 2, 3, 5, 6] for idx in sampled
        ), f"All sampled frames must be in pool [0, 1, 2, 3, 5, 6], got {sampled}"

    def test_last_strategy(self, sampler_with_frames):
        """Test 'last' strategy selects most recent frames."""
        result = sampler_with_frames.framesFor(
            mainInd=8, samples=[0, 1, 2, 3, 5, 6, 7], steps=4, stepsSampling="last"
        )

        # Verify frame selection validity
        assert (
            len(result) == 4
        ), f"Expected 4 frames for 'last' sampling, got {len(result)} frames: {result}"
        assert 8 in result, f"mainInd 8 should be included in result, but got {result}"
        assert result == sorted(result), f"Result should be sorted, but got {result}"
        assert len(result) == len(set(result)), f"Result has duplicate frames: {result}"

        # Verify exact frames: most recent 3 + mainInd
        expected = sorted([5, 6, 7, 8])
        assert result == expected, f"Expected {expected}, got {result}"

    def test_last_strategy_exact_values(self, sampler_with_frames):
        """Test 'last' strategy returns exactly expected indices."""
        result = sampler_with_frames.framesFor(
            mainInd=5, samples=[0, 1, 2, 3, 4, 6, 7, 8], steps=3, stepsSampling="last"
        )

        # Verify frame selection validity
        assert (
            len(result) == 3
        ), f"Expected 3 frames for 'last' sampling, got {len(result)} frames: {result}"
        assert 5 in result, f"mainInd 5 should be included in result, but got {result}"
        assert result == sorted(result), f"Result should be sorted, but got {result}"
        assert len(result) == len(set(result)), f"Result has duplicate frames: {result}"

        # Most recent 2 from samples [0,1,2,3,4,6,7,8] + mainInd=5 → [5, 7, 8]
        assert result == [5, 7, 8], f"Expected [5, 7, 8], got {result}"

    def test_dict_strategy(self, sampler_with_frames):
        """Test dict strategy with max frames parameter."""
        result = sampler_with_frames.framesFor(
            mainInd=7,
            samples=[0, 1, 2, 3, 5, 6],
            steps=4,
            stepsSampling={"max frames": 3},
        )

        # Verify frame selection validity
        assert (
            len(result) == 4
        ), f"Expected 4 frames for dict sampling with max_frames=3, got {len(result)} frames: {result}"
        assert 7 in result, f"mainInd 7 should be included in result, but got {result}"
        assert result == sorted(result), f"Result should be sorted, but got {result}"
        assert len(result) == len(set(result)), f"Result has duplicate frames: {result}"

        # All frames should be from pool or mainInd
        assert all(
            idx == 7 or idx in [0, 1, 2, 3, 5, 6] for idx in result
        ), f"Invalid frames in result {result}, expected mainInd=7 or samples [0, 1, 2, 3, 5, 6]"

    def test_dict_strategy_various_max_frames(self, sampler_with_frames):
        """Test dict strategy respects different max frames values."""
        samples = [0, 1, 2, 3, 4, 6, 7, 8]
        for max_frames in [1, 2, 3, 5]:
            result = sampler_with_frames.framesFor(
                mainInd=9,
                samples=samples,
                steps=4,
                stepsSampling={"max frames": max_frames},
            )

            # Verify frame selection validity
            assert (
                len(result) == 4
            ), f"dict(max={max_frames}): expected 4 frames, got {len(result)} frames: {result}"
            assert (
                9 in result
            ), f"dict(max={max_frames}): mainInd 9 should be included in result, but got {result}"
            assert result == sorted(
                result
            ), f"dict(max={max_frames}): result should be sorted, but got {result}"
            assert len(result) == len(
                set(result)
            ), f"dict(max={max_frames}): result has duplicate frames: {result}"

    def test_uniform_different_step_counts(self, sampler_with_frames):
        """Test uniform strategy with various step counts."""
        samples = [0, 1, 2, 3, 4, 6, 7, 8]
        for steps in [2, 3, 4, 5]:
            result = sampler_with_frames.framesFor(
                mainInd=9, samples=samples, steps=steps, stepsSampling="uniform"
            )

            # Verify frame selection validity
            assert (
                len(result) == steps
            ), f"uniform(steps={steps}): expected {steps} frames, got {len(result)} frames: {result}"
            assert (
                9 in result
            ), f"uniform(steps={steps}): mainInd 9 should be included in result, but got {result}"
            assert result == sorted(
                result
            ), f"uniform(steps={steps}): result should be sorted, but got {result}"
            assert len(result) == len(
                set(result)
            ), f"uniform(steps={steps}): result has duplicate frames: {result}"

    def test_strategies_with_various_main_indices(self, sampler_with_frames):
        """Test uniform and last strategies with different mainInd positions."""
        for mainInd in [1, 4, 7]:
            samples = [i for i in range(10) if i != mainInd]
            for strategy in ["uniform", "last"]:
                result = sampler_with_frames.framesFor(
                    mainInd=mainInd, samples=samples, steps=5, stepsSampling=strategy
                )

                # Verify frame selection validity
                assert (
                    len(result) == 5
                ), f"{strategy}(main={mainInd}): expected 5 frames, got {len(result)} frames: {result}"
                assert (
                    mainInd in result
                ), f"{strategy}(main={mainInd}): mainInd {mainInd} should be included in result, but got {result}"
                assert result == sorted(
                    result
                ), f"{strategy}(main={mainInd}): result should be sorted, but got {result}"
                assert len(result) == len(
                    set(result)
                ), f"{strategy}(main={mainInd}): result has duplicate frames: {result}"

    def test_unsorted_samples_list(self, sampler_with_frames):
        """Test strategies handle unsorted samples input correctly."""
        unsorted_samples = [2, 7, 1, 9, 3, 6]
        for strategy in ["uniform", "last"]:
            result = sampler_with_frames.framesFor(
                mainInd=5, samples=unsorted_samples, steps=4, stepsSampling=strategy
            )

            # Verify frame selection validity
            assert (
                len(result) == 4
            ), f"{strategy}(unsorted): expected 4 frames, got {len(result)} frames: {result}"
            assert (
                5 in result
            ), f"{strategy}(unsorted): mainInd 5 should be included in result, but got {result}"
            assert result == sorted(
                result
            ), f"{strategy}(unsorted): result should be sorted, but got {result}"
            assert len(result) == len(
                set(result)
            ), f"{strategy}(unsorted): result has duplicate frames: {result}"
