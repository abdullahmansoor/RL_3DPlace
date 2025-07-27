# RL_3DPlace
RL_3DPlace is a reinforcement learning-based placement framework for monolithic 3D integrated circuits (ICs). It leverages the Poetry package manager for dependency management and execution.

## 📌 Features
Poetry-based setup for dependency management
Reinforcement Learning (RL) approach for 3D IC placement
Benchmark support for various designs
Customizable execution for specific designs

## 🚀 Installation
1. Clone the Repository
```
git clone https://github.com/abdullahmansoor/RL_3DPlace.git
cd RL_3DPlace
```

3. Install Dependencies using Poetry
Ensure you have Poetry installed. If not, install it via:
```
pip install poetry
```
Then, install dependencies:
```
poetry install
```

## 🏃Running the Project

1. Run Default Placement Flow
```   
poetry run python src/rl_3dplace/DHARL_flow.py
```
2. Run Placement for a Specific Design
```   
poetry run python src/rl_3dplace/DHARL_flow.py -designName \<designName\>
```
3. Evaluate Placement cost function
```
poetry run  python src/rl_3dplace/design_engines/eval_placement_cost.py -designName <designName e.g muxshifter8> -inputPlacementFile <3d placement e.g. muxshifter8_3d.pl>
```

Available Benchmarks
The project supports the following benchmark designs:
- rlcase1
- muxshifter4
- muxshifter8
- muxshifter16
- muxshifter16b
- muxshifter32
- muxshifter64
- muxshifter128
- picorv32a
- tate
- jpeg

## 🏗️ Project Structure:
```
RL_3DPlace/
│── src/
│   ├── rl_3dplace/           # Main directory containing the placement script
│   ├── PDLibs/               # Legacy code mainly for Bookshelf format support
│── data/
│   ├── benchmarks/           # Directory containing supported benchmark designs
│   ├── pagn_models/          # Graph ML models for placement (some models are too large for GitHub)
│   ├── rlagent_models/       # RL Agent policy models
│── pyproject.toml            # Poetry dependency configuration
│── README.md                 # Project documentation
```



## 🤝 Contribution:
We welcome contributions! To contribute:
- Fork the repository.
- Create a new branch (feature-xyz).
- Commit changes and submit a pull request.

## 📜 License
This project is licensed under the MIT License.
