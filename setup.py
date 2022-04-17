from setuptools import setup, find_packages

setup(
    name="peer_to_peer_lending",
    packages=find_packages(),
    package_data={"data": ["dummy_data/dummy_listings.csv", "dummy_data/dummy_loans.csv"]},
    include_package_data=True,
)
