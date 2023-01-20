[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9521438&assignment_repo_type=AssignmentRepo)
# Data Acquisition and Processing Systems (DAPS) (ELEC0136)

# The previous files are submitted wrongly to another repository where can find all the GitHub Commit Histories: https://github.com/mujiexu2/ELEC0136-Assignment-SN19027325


Welcome to the final assignment of the _Data Acquisition and Processing Systems_ (DAPS) course ELEC0136 at UCL.

This repository contains the scaffolding for submitting your code.
*Please work only on the `main` branch.*
A quick reminder that the criteria for grading your code will be based on the following:

1. **Reproducibility.** We run an automated test suite on your code to make sure it runs. Please, do **NOT** merge your `feedback` pull request, or manually run the test suite on GitHub. The test suite will run automatically when we merge your `feedback` pull request. Do not call methods that require user input, included plotting methods that require exiting. The procedure performs two operations:
   1. Creates a fresh conda environment from a `environment.yml` file in the parent folder of the repo. The file must follow the format described in the [conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). The procedure runs something close to: `conda env create -f environment.yml`.
   2. Executes the code in the `main.py` file, something close to: `python main.py`.
2. **Documentation.** We will look at the docstrings of your functions and classes. Make sure the docstrings are clear and complete, and avoid a single sentence that only summarises the method signature. Prefer, instead, to explain the purpose of the function, the parameters, and the return value, and clarify any ambiguity that may arise from the variable or method names, e.g., _`data` is a `pandas.DataFrame` containing the data to be processed_. The docstrings should follow the Google style, as described [here](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) and [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
3. **Quality.** The code should respect the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide. We will use [pylint](https://www.pylint.org/) to check the code quality. You can run it locally with `pylint main.py` or `pylint src/` to check the whole codebase. You can also use the `pylint` extension in VSCode. The results of the test suite will be available in the `feedback` pull request, and represent only a guide for the quality score, which will be instead assigned by the teaching staff.
4. **Modularity/Organisation.** The code should be split into multiple files and functions. The `main.py` file should be short and should only contain the code to run the experiments. The `src/` folder should contain the code to process the data, train the model, and generate the visualisations. The `src/` folder should contain a `__init__.py` file to make it a Python module.
5. **Version controlled.** The code should be version-controlled using git. Prefer modular commits to a single commit with all the code. The commit messages should be clear and concise. The repository should also contain a `README.md` file with a description of the project and instructions on how to run the code.

The curious ones can check the `.github/workflows/score.yml` file to see what the automatic checks will do.

### Submitting your assignment

To submit your assignment, please comment on the `feedback` PR with `@UCL-ELEC0136/teaching, I submit!`. This way we will receive the notification you submitted and know that the assignment is complete.
If you submit after the deadline, every commit with timestamp later than the deadline will not be considered.

### Reporting bugs
Should you find a bug in the assignment procedure, the code, or the assignment text, please reach out by email, or by opening an issue on the repository. We will try to fix it as soon as possible.


GLHF! :smile:
