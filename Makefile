help::        ## to print all the available targets
    @echo "\nAvailable targets:\n"
    @grep -E '^[a-zA-Z_-]+:.?## .$$' $(MAKEFILE_LIST) | tr ':' ' ' ; echo

install::    ## to install requirements
    pip install .

format::    ## to format the code with the ruff tool
    ruff format ml_owls tests utils

format-check::    ## to check the formatting code with ruff
    ruff format --check ml_owls tests utils

lint::        ## to check the code style
    ruff check ml_owls tests utils

lint-fix::    ## to check and fix the code style
    ruff check --fix ml_owls tests utils

test::        ## to launch the tests
    pytest -v --doctest-modules tests
