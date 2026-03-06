from setuptools import setup

package_name = 'pointcloud_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='PointCloud2 publisher',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'publisher = pointcloud_pub.processing_node:main',

        ],
    },
)