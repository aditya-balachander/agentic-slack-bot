# Agentic Slack Bot

## Environment Setup

To get started with this project, follow these steps:

1.  **Create a Virtual Environment:**

    ```bash
    python3.11 -m venv .venv
    ```

    This command creates a virtual environment named `.venv` in your project directory.

2.  **Activate the Virtual Environment:**

    - **On macOS and Linux:**

      ```bash
      source .venv/bin/activate
      ```

      Once activated, you should see `(.venv)` at the beginning of your terminal prompt.

3.  **Install Poetry:**

    ```bash
    pip install poetry
    ```

4.  **Install Project Dependencies:**

    ```bash
    poetry install
    ```

    Poetry will read the `pyproject.toml` and `poetry.lock` files to install the exact versions of the dependencies required for this project within your active virtual environment.

## Managing Dependencies

This project uses Poetry to manage its dependencies. Here's how to handle common dependency-related tasks:

### Adding New Dependencies

If you need to add a new dependency to the project:

1.  **Use the `poetry add` command:**
    Open your terminal, ensure your virtual environment is activated, and navigate to the project's root directory. Then, run the following command, replacing `<package-name>` with the name of the package you want to add:

    ```bash
    poetry add <package-name>
    ```

    You can also specify a version constraint if needed:

    ```bash
    poetry add <package-name>@^1.2.3
    ```

2.  **Poetry will update `pyproject.toml` and `poetry.lock`:**
    After successfully adding the dependency, Poetry will automatically update the `pyproject.toml` file with the new dependency and its version constraint. It will also update the `poetry.lock` file to reflect the exact versions of all dependencies, including the newly added one and its transitive dependencies.

### Updating Existing Dependencies

To update dependencies to their latest compatible versions:

1.  **Use the `poetry update` command:**
    In your terminal, with the virtual environment activated and in the project's root directory, run:

    ```bash
    poetry update
    ```

    This command will look for newer compatible versions of your dependencies as specified in `pyproject.toml` and update `poetry.lock` accordingly.

2.  **Update specific dependencies:**
    To update specific packages, you can name them:

    ```bash
    poetry update <package-name1> <package-name2>
    ```

### Updating the Lock File

If you've manually edited the `pyproject.toml` file (e.g., changed version constraints), you should run the `poetry lock` command to update the `poetry.lock` file to reflect these changes:

```bash
poetry lock
```
