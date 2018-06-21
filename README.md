# eddytools

[![PyPI](https://img.shields.io/pypi/v/eddytools.svg)](https://pypi.org/project/eddytools/) [![Build Status](https://travis-ci.org/edugonza/eddytools.svg?branch=master)](https://travis-ci.org/edugonza/eddytools) [![Coverage Status](https://img.shields.io/coveralls/github/edugonza/eddytools/master.svg)](https://coveralls.io/github/edugonza/eddytools?branch=master)

Event Data Discovery Tools from databases. This project includes a generic importer from databases to OpenSLEX and event discovery features.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. To use the library as it is,
 go to [Installing](#installing). For development and testing purposes, go to [Running the tests](#running-the-tests).
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python>=3.6

### Installing

The library can be installed using pip:

```
pip install eddytools
```

To run the code from the sources (cloning the repository) for development purposes, first you will have to install
the dependencies:

```
pip install -r -U requirements.txt
```

## Running the tests

The unit tests will validate, among other things, the extraction of data from a database and the
transformation to the OpenSLEX format. In order to test this functionality, a postgresql docker
instance with a database is required. Assuming that docker is installed and running, execute the following commands from
the root directory of the project.

```
cd data/ds2
sh build-image.sh
sh run-image.sh
sh check-health.sh
```

The last command with return when the database container is up, running and ready to receive incoming connections.

Also, the test dependencies must be satisfied:

```
pip install -r -U requirements_test.txt
```

And finally, we execute the tests with:

```
python setup.py test
```

To gather code coverage data, the previous command can be substituted by:

```
coverage run --source=eddytools setup.py test
```

To report on the results:

```
coverage report
```

And to generate the html output:

```
coverage html
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/edugonza/eddytools/tags).

## Authors

* **Eduardo Gonzalez Lopez de Murillas** - *Schema discovery - Case Notion Discovery - Maintainer* - [edugonza](https://github.com/edugonza)
* **Roy Wolters** - *Initial work on extraction - Event data detection* - [roywolters](https://github.com/roywolters)

See also the list of [contributors](https://github.com/edugonza/eddytools/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
