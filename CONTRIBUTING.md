Contributing to Matchzoo
----------

Welcome! Matchzoo is a community project that aims to work for a wide range of NLP and IR tasks such as Question Answering, Information Retrieval, Paraphrase identification etc. Your experience and what you can contribute are important to the project's success.

> Note: Matchzoo is developed under Python 3.6.

Discussion
----------

If you've run into behavior in Matchzoo you don't understand, or you're having trouble working out a good way to apply it to your code, or you've found a bug or would like a feature it doesn't have, we want to hear from you!

Our main forum for discussion is the project's [GitHub issue tracker](https://github.com/NTMC-Community/MatchZoo/issues).  This is the right place to start a discussion of any of the above or most any other topic concerning the project.

For less formal discussion we have a chat room on Wechat (mostly Chinese speakers). Matchzoo core developers are almost always present; feel free to find us there and we're happy to chat. Please add *CLJ_Keep* as your wechat friend, she will invite you to join the chat room.

First Time Contributors
-----------------------

Matchzoo appreciates your contribution! If you are interested in helping improve Matchzoo, there are several ways to get started:

* Work on [new models](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match).
* Work on [tutorials](https://github.com/NTMC-Community/MatchZoo/tree/master/tutorials).
* Work on [documentation](https://github.com/NTMC-Community/MatchZoo/tree/master/docs).
* Ask on [the issue tracker](https://github.com/NTMC-Community/MatchZoo/issues) about good beginner issues.

Submitting Changes
------------------

Even more excellent than a good bug report is a fix for a bug, or the implementation of a much-needed new model. 

(*)  We'd love to have your contributions.

(*) If your new feature will be a lot of work, we recommend talking to us early -- see below.

We use the usual GitHub pull-request flow, which may be familiar to you if you've contributed to other projects on GitHub -- see blow. 

Anyone interested in Matchzoo may review your code.  One of the Matchzoo core developers will merge your pull request when they think it's ready.
For every pull request, we aim to promptly either merge it or say why it's not yet ready; if you go a few days without a reply, please feel
free to ping the thread by adding a new comment.

For a list of Matchzoo core developers, see [Readme](https://github.com/NTMC-Community/MatchZoo/blob/master/README.md).

Contributing Flow
------------------

1. Fork the latest version of [MatchZoo](https://github.com/NTMC-Community/MatchZoo) into your repo.
2. Create an issue under [NTMC-Community/Matchzoo](https://github.com/NTMC-Community/MatchZoo/issues), write description about the bug/enhancement.
3. Clone your forked MatchZoo into your machine, add your changes together with associated tests.
4. Run `make test` with terminal, ensure all unit tests & integration tests passed on your computer.
5. Push to your forked repo, then send the pull request to the official repo. In pull request, you need to create a link to the issue you created using `#[issue_id]`, and describe what has been changed.
6. Wait [continuous integration](https://travis-ci.org/faneshion/MatchZoo/) passed.
7. Wait [Codecov](https://codecov.io/gh/faneshion/MatchZoo) generate the coverage report.
8. We'll assign reviewers to review your code.


Your PR will be merged if:
- Funcitonally benefit for the project.
- Passed Countinuous Integration (all unit tests, integration tests and [PEP8](https://www.python.org/dev/peps/pep-0008/) check passed).
- Test coverage didn't decreased, we use [pytest](https://docs.pytest.org/en/latest/).
- With proper docstrings, see codebase as examples.
- With type hints, see [typing](https://docs.python.org/3/library/typing.html). 
- All reviewers approved your changes.


**Thanks and let's improve MatchZoo together!**