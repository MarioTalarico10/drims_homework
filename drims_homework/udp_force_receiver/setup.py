from setuptools import setup
from glob import glob
import os

package_name = 'udp_force_receiver'

setup(
    name=package_name,
    version='0.0.0',

    packages=[package_name],

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='mario',
    maintainer_email='mario.talarico@polimi.it',

    description='UDP receiver for force data',

    license='Politecnico di Milano',

    entry_points={
        'console_scripts': [
            'udp_force_node = udp_force_receiver.udp_force_node:main',
        ],
    },
)