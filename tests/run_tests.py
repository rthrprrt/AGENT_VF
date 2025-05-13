# tests/run_tests.py
# -*- coding: utf-8 -*-
import pytest
import typer
from typing_extensions import Annotated
import sys # Import sys to check platform

app = typer.Typer()

@app.command()
def main(
    marker: Annotated[str, typer.Option(help="Run tests with a specific marker (e.g., 'unit', 'integration').")] = None,
    keyword: Annotated[str, typer.Option("-k", help="Run tests matching the keyword expression.")] = "",
    verbose: Annotated[bool, typer.Option("-v", help="Enable verbose output.")] = False,
    exit_first: Annotated[bool, typer.Option("-x", "--exitfirst", help="Exit instantly on first error or failed test.")] = False,
):
    """
    Runs the pytest test suite for the AGENT_VF project.

    Allows filtering by markers (unit, integration) and keywords.
    """
    args = []
    if marker:
        print(f"Running tests marked as: {marker}")
        args.extend(["-m", marker])
    else:
        print("Running all tests (unit and integration)")

    if keyword:
        print(f"Filtering tests with keyword: {keyword}")
        args.extend(["-k", keyword])

    if verbose:
        args.append("-v")

    if exit_first:
        args.append("-x")

    # Add the tests directory to the arguments
    args.append("tests/")

    print(f"\nRunning pytest with arguments: {args}\n")
    exit_code = pytest.main(args)

    # Correction: Use simple ASCII characters for status messages
    if exit_code == 0:
        print("\n[OK] All selected tests passed!")
    elif exit_code == pytest.ExitCode.NO_TESTS_COLLECTED:
         print("\n[WARN] No tests were collected (check markers/keywords).")
    else:
        print(f"\n[FAIL] Some tests failed (exit code: {exit_code})")

    raise typer.Exit(code=exit_code)

if __name__ == "__main__":
    # Optional: Force UTF-8 encoding for stdout on Windows if needed,
    # though avoiding emojis is safer.
    # if sys.platform == "win32":
    #     try:
    #         sys.stdout.reconfigure(encoding='utf-8')
    #         sys.stderr.reconfigure(encoding='utf-8')
    #     except Exception as e:
    #         print(f"Warning: Could not reconfigure stdout/stderr to UTF-8: {e}")
    app()

# End of file