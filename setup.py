from setuptools import setup
import os
from glob import glob

package_name = 'dextel'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'assets'), glob('dextel/assets/*')),
        (os.path.join('share', package_name, 'assets/meshes'), glob('dextel/assets/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='DexTel Hand Tracking Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur3_realsense_hamer = dextel.ur3_realsense_hamer:main',
            'dextel_node = dextel.dextel_node:main',
            'simple_robotiq_driver = dextel.simple_robotiq_driver:main',
        ],
    },
)
