classdef TestEliminateBlinkingDeLoc15MCMC < matlab.unittest.TestCase
    methods (Test)
        function testAllNonBlinks(testCase)
            Locs = rand(20, 2) * 100;
            Frames = 1:20;
            Resolution = 1;
            A = 1;
            Deviation = ones(10, 10) * 0.4;
            Traj = 1:20;

            [Loc, Probs] = Eliminate_Blinking_De_Loc15_MCMC(Locs, Frames, Resolution, A, Deviation, Traj);
            testCase.verifyEqual(sum(Loc == 1), 20);
        end

        function testAllBlinks(testCase)
            Locs = [(0:19)', (0:19)'];
            Frames = 0:19;
            Resolution = 1;
            A = 2;
            Deviation = ones(10, 10) * 0.9;
            Traj = 1:20;

            [Loc, Probs] = Eliminate_Blinking_De_Loc15_MCMC(Locs, Frames, Resolution, A, Deviation, Traj);
            testCase.verifyGreaterThan(sum(Loc == 0), 0);
        end

        function testRandomMix(testCase)
            rng(0);  % For reproducibility
            Locs = rand(20, 2) * 10;
            Frames = randi([0, 5], 1, 20);
            Resolution = 1;
            A = 3;
            Deviation = rand(10, 10);
            Traj = 1:20;

            [Loc, Probs] = Eliminate_Blinking_De_Loc15_MCMC(Locs, Frames, Resolution, A, Deviation, Traj);
            testCase.verifyEqual(length(Loc), 20);
            testCase.verifyTrue(all(Loc == 0 | Loc == 1));
        end
    end
end

runtests('TestEliminateBlinkingDeLoc15MCMC')
