from setuptools import setup

setup(
        name='eddytools',
        use_scm_version=True,
        description='Event Data Discovery tool',
        url='https://github.com/edugonza/eddytools',
        author='Eduardo Gonzalez Lopez de Murillas',
        author_email='edu.gonza.lopez@gmail.com',
        keywords='process mining events data extraction databases openslex',
        license='MIT',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
        ],
        setup_requires=['pytest-runner','setuptools_scm'],
        tests_require=['pytest'],
        install_requires=['sqlalchemy','setuptools','psycopg2'],
        python_requires='>=3.6',
        packages = ['eddytools'],
        package_data={'eddytools': ['resources']},
        include_package_data=True,
)
