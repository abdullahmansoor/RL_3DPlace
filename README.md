# RL_3DPlace
RL_3DPlace is a reinforcement learning-based placement framework for monolithic 3D integrated circuits (ICs). It leverages the Poetry package manager for dependency management and execution.

📌 Features
Poetry-based setup for dependency management
Reinforcement Learning (RL) approach for 3D IC placement
Benchmark support for various designs
Customizable execution for specific designs

🚀 Installation
1. Clone the Repository
git clone https://github.com/abdullahmansoor/RL_3DPlace.git
cd RL_3DPlace


2. Install Dependencies using Poetry
Ensure you have Poetry installed. If not, install it via:

pip install poetry

Then, install dependencies:
poetry install



🏃Running the Project
1. Run Default Placement Flow
poetry run python src/rl_3dplace/DHARL_flow.py

2. Run Placement for a Specific Design
poetry run python src/rl_3dplace/DHARL_flow.py -designName <designName>

Available Benchmarks
The project supports the following benchmark designs:
rlcase1
muxshifter4
muxshifter8
muxshifter16
muxshifter16b
muxshifter32
muxshifter64
muxshifter128
picorv32a

Project Structure
src/rl_3dplace/ – Main directory containing the placement script.
src/PDLibs - Legacy code mainly for bookshelf format support
data/benchmarks/ – Directory containing supported benchmark designs.
data/pagn_models/ - Graph ML models for the above designs. Some models with large files can't be uploaded on github.com
data/rlagent_models/ - RL Agent Policy models
pyproject.toml – Poetry dependency configuration.
README.md – Project documentation.


Contribution
We welcome contributions! To contribute:

Fork the repository.
-Create a new branch (feature-xyz).
-Commit changes and submit a pull request.

License
This project is licensed under the MIT License.
