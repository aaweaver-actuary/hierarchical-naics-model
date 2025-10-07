# To-Do List for Hierarchical NAICS Model Dashboard Enhancements

## Overview 

Your job is to inVdependently work based on this fully test-covered codebase, finishing a previously-started fully-featured model fit dashboard that can be used to evaluate convergence and predictive power for a bayesian logistic regression model that features a highly-hierarchical structure for naics codes and zip codes.

Please run `make test` whenever needed to get the current list of lint, type checking, unit testing, and test coverage errors, and try to resolve them step by step. After each update, you should run make test again to verify that your update had the intended effect, and either way, use the feedback you get to iterate further.

You are authorized to work independently as long as you follow this incremental refactoring approach.

I attached the `README.md` file for this repo. You should first of all familiarize yourself with its content. After that you should proceed to create a fulll end-to-end test of this project that:

At each stage, you need to use strict test-driven development with a red-green-refactor flow. You must OBSERVE a failing test before writing production code. You must ONLY write enough prod code to satisfy that test.

You should first start by defining test stubs, then ignoring most of them and work test by test.

You must only use make test to run the testing suite. You are authorized to do this as much as needed. You are not authorized to run pytest directly.

Please make a detailed todo list, and use your available tools to fetch api documentation and search for code examples as needed.

Think carefully about:

1. What framework to use to accomplish this
2. Best practices for visualizing, validating, and observing Bayesian models like this
3. How to visualize hierarchical variables
4. What interactivity will enhance the usefulness of the dashboard

You must ensure you are still following the same TDD red-green-refactor protocol, and linting and testing at each step.

You must also strive to keep high-level logic straightforward and focused on orchestration of lower-level helpers.

[x] Full trace plots with unified hover
    [x] Add failing test to confirm full trace plot presence and structure in API response
    [x] Ensure a test fails if the number of displayed draws is lower than the total number of draws
    [x] Implement the code to generate and include the full trace plot in the response.
    [x] Confirm the test passes after implementation.
    [x] Ensure a test fails if the hover does not show all chains' values at a given draw, and focuses only on the draw closest to the cursor.
    [x] Implement the code to unify hover across chains in the trace plot.
    [x] Confirm the test passes after implementation.
[x] Autocorrelation diagnostics
    [x] Add failing test to confirm autocorrelation diagnostic presence and structure in API response
    [x] Implement the code to compute and include the autocorrelation diagnostics in the response.
    [x] Confirm the test passes after implementation.
    [x] Ensure a test fails if the autocorrelation plot does not show lags up to at least 100 or 10% of total draws, whichever is lower.
    [x] Implement the code to ensure the autocorrelation plot shows lags up to at least 100 or 10% of total draws, whichever is lower.
    [x] Confirm the test passes after implementation.
    [x] Ensure a test fails if there is no interpretation guidance for the autocorrelation plot.
    [x] Implement the code to add interpretation guidance for the autocorrelation plot.
    [x] Confirm the test passes after implementation.
[x] Posterior predictive check (PPC) KDE overlays
    [x] Add failing test to confirm PPC KDE overlay presence and structure in API response
    [x] Implement the code to compute and include the PPC KDE overlay in the response.
    [x] Confirm the test passes after implementation.
    [x] Ensure a test fails if the PPC KDE overlay does not match the observed data histogram in terms of x-axis range and scale.
    [x] Implement the code to ensure the PPC KDE overlay matches the observed data histogram in terms of x-axis range and scale.
    [x] Confirm the test passes after implementation.

[x] Dashboard variable dropdown follow-up
    [x] Variable dropdown should show uncluttered base names with a secondary selector for indices: Done (tests + implementation).
[x] Prior density overlay QA
    [x] Add failing test to confirm prior curve presence and structure in API response
    [x] Observe the failure, then implement the code to generate and include the prior density curve in the response.
    [x] Confirm the test passes after implementation.
[x] Convergence indicator badges for R-hat / ESS
    [x] Add failing test for presence and thresholds of R-hat in API response
     [x] Implement the logic to compute and include this diagnostic in the response
    [x] Confirm the test passes after implementation.
    [x] Add failing test for presence and thresholds of ESS in API response
    [x] Implement the logic to compute and include this diagnostic in the response
    [x] Confirm the test passes after implementation.
[x] Summary chips for key posterior stats
    [x] Add failing test for presence and correctness of mean, median, 5/95% quantiles in API response
    [x] Implement the logic to compute and include these stats in the response
    [x] Confirm the test passes after implementation.
[ ] Trace plot smoothing and draw downsampling
    [ ] Add failing test for trace plot smoothing and draw downsampling in API response
    [ ] Implement the logic to apply smoothing and downsampling to the trace data
    [ ] Confirm the test passes after implementation.
[ ] Posterior predictive check (PPC) KDE overlays
    [ ] Add failing test for presence and structure of PPC KDE overlay data in API response
    [ ] Implement the logic to compute and include the PPC KDE overlay data in the response
    [ ] Confirm the test passes after implementation.
[ ] Chain visibility toggles on trace plot
[ ] Prior overlay toggle on posterior histogram
[ ] Posterior predictive check (PPC) overlay toggle on observed data histogram
[ ] Posterior predictive check (PPC) overlay toggle on density plot
[ ] Density plot percentile shading / CI ribbons
[ ] Density plot observed data rug
[ ] Density plot PPC overlay
[ ] Density plot log-scale toggle
[ ] Density plot bandwidth slider
[ ] Hierarchical grouping in variable dropdown
[ ] Hierarchical group tagging in variable sidebar
[ ] Posterior sample download (CSV/JSON)
    [ ] Add failing test for presence and structure of posterior sample download in API response
    [ ] Implement the logic to generate and include the posterior sample download data in the response
    [ ] Confirm the test passes after implementation.
[ ] Reliability plot tooltips and accessible annotations
    [ ] Reliability plot log-scale toggle
    [ ] Reliability plot bin count slider
    [ ] Lift badge and tooltip
[ ] Ranking plot with lift badge
    [ ] Ranking plot highlight for best k% and lift badge
[ ] Calibration metrics table beneath the plot
[ ] Inference presets (common NAICS/ZIP combos)
[ ] Inference result export / copy-to-clipboard
[ ] Decision-flow table sorting and totals row
[ ] Dashboard loading states / skeleton for heavy tabs
[ ] Keyboard navigation & ARIA polish across tabs
[ ] Automated dashboard screenshot smoke (playwright/png)
[ ] README / docs refresh covering the new dashboard features
[ ] Extend the dashboard tests to capture the new Variable tab UX (grouped base [ ] names with index selectors) before implementing the UI/payload changes.
[ ] Iteratively add tests + code for the remaining Variable tab improvements: prior overlays, full traces with unified hover, autocorrelation diagnostics, and PPC KDE overlays.
[ ] Move on to the Validation tab, expanding tests to cover interpretive annotations and then updating the Plotly logic accordingly.
[ ] Revisit synthetic data generation to introduce richer noise and stronger hierarchy signals, accompanied by parameter-recovery assertions and contribution checks.
[ ] Research hierarchy visualization best practices, summarize actionable insights, and translate them into dashboard enhancements or follow-on tasks.

You should first start by defining test stubs, then ignoring most of them and work test by test.

You must ensure you are still following the same TDD red-green-refactor protocol, and linting and testing at each step.

You must also strive to keep high-level logic straightforward and focused on orchestration of lower-level helpers. Private low-level helpers should be well-tested, and each should be focused on a single task, and no more than 5 lines long. Use long, highly-expressive names for these low-level helpers, and keep them private to the module unless you have a very good reason to expose them.