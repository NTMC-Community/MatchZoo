HOW TO SEND A PULL REQUEST? MATCHZOO'S WORK FLOW
==========

For users:

1. Fork the latest version of MatchZoo into your repo.
2. Create an issue under faneshion/Matchzoo, write the description of the bug/enhancement.
3. Clone your forked MatchZoo into your machine, add your changes together with associated unit test.
4. Run `make init` and `make test`  using terminal/command line, ensure dependency check & unit tests passed on your computer.
5. Push to your forked repo, send the pull request, in PR, you need to create a link to the issue you created using `#[issue_id]`, and describe what has been changed.
6. Wait CI passed.
7. Wait Codecov generate the coverage report.
8. We'll assign reviewers to review your code.
9. Your PR will be merged if:
    - Funcitonally benefit for the project.
    - Passed Countinuous Integration (all unit-test and PEP8 check passed).
    - Coverage didn't decreased (we use pytest).
    - All reviewers has approved.



For code developers:

1. Create a sub-branch under `2.0`, naming conventions:
    + For new features or enhancements, use `feature/[branch-name]`.
    + For bugs, use `hotfix/[branch-name]`
2. Create an issue under faneshion/Matchzoo, write the description of the bug/enhancement.
3. `git pull` to get the latest code, `git checkout feature/[branch-name]` to switch to your newly created branch.
4. Edit code & add unit test.
5. Run `make init` and `make test` to ensure all the dependencies & unit tests has been passed on your local machine.
6. Create a new PR, assign yourself as `assignees`, assign 1 or 2 `reviewers` to review your code.
7. Create a link between your PR & Issue using `#[issue-id]`, write descriptions of what has changed.
8. Wait  CI passed
9. Wait Codecov generates the coverage report, and ensure coverage didn't decrease.
9. Wait all reviewers agreed this piece of code can be merged into the code base.
10.  IMPORTANT: USE `SQUASH AND MERGE` INSTEAD OF `MERGE`!
11. Close the issue you created.
12. Remove sub-branch you created for this issue (i.e. `feature/branch-name` or `hotfix/branch-name`) both locally and remotely.