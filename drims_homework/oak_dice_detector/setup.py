from setuptools import (
    find_packages,
    setup
)

import os
from glob import glob


package_name = 'oak_dice_detector'


setup(

    name=package_name,

    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[

        (
            'share/ament_index/resource_index/packages',
            [
                'resource/' +
                package_name
            ]
        ),

        (
            'share/' +
            package_name,
            [
                'package.xml'
            ]
        ),

        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob(
                'launch/*.launch.py'
            )
        ),

        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            glob(
                'config/*.yaml'
            )
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='Mario',

    maintainer_email='mario.talarico@polimi.it',

    description=(
        'OAK camera-based dice '
        'detection and face recognition'
    ),

    license='Giovanni Indino Pornhub',

    tests_require=[
        'pytest'
    ],

    entry_points={

        'console_scripts': [

            'dice_detector_node = '
            'oak_dice_detector.'
            'dice_detector_node:main',

        ],
    },
)