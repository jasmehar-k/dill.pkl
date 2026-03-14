# dill.pkl

<<<<<<< HEAD
Project layout:

- `front end/` contains the React + Vite frontend
- the repo root is available for the Python backend

To run the frontend:

```bash
cd "front end"
npm install
npm run dev
```
=======
## Inspiration

## What it does

## How we built it

## Challenges we ran into

## Accomplishments that we're proud of

## What we learned

## What's next for dill.pkl
>>>>>>> 2bb71e5592582795f0a7f03696a929a5f191a74d


## Agents
### Master orchestrator agent
what it does:
- reads user input
- decide workflows based on data characteristics
- coordinates all other agents
- makes go/no-go decisions at each step

example output:
"data is 10k rows, 50 features, class imbalance 10:1
-> recommended: aggressive preprocessing + tree-based models + SMOTE for balance"

### Data analyzer agent
what it does:
- decides preprocessing strategy based on analysis
- handles missing values intelligently 
- scales/treansforms appropriately
- manage categorical encoding
decision examples:
- "high missing % -> use KNN imputation"
- "skewed distribution -> log transform:
- "many categories -> target encode"

### Feature engineering agent
what it does:
- decides which features to create
- creates polynomial/interaction features
- removes redundant features
- optimize feature count
reasoning:
- "already 500 features -> don't add more"
- "weak linear relationship -> create polynomial"
- "high dimensionality -> use PCA"

### Model selection agent
what it does:
- recommends algorithms based on data
- explains why each model is chosen 
- suggests hyperparameter ranges
logic:
- "non-linear patterns -> use tree models"
- "high-dimensional sparse -> use linear"
- "imbalanced -> use XGBoost with weights"

### Training 
what it does:
- manages model training
- performs hyperparameter tuning
- monitors for overfitting
- manages computational resources
capabilities:
- decides tunng budget per model
- uses Bayesian/random search
- early stopping detection
- cross-validation handling 

### Evaluation agent
what it does:
- tests models on validation/test sets
- calculates appropriate metrics
- detects overfitting/underfitting
- makes deployment decision
decisions:
- "performance good, gap acceptable -> deploy"
- "overfit detected -> iterate"
- "performance poor -> reject"

### Explanation generator
what it does:
- creates human-readable explanations 
- SHAP/LIME for feature importance
- generates reports
output:
- feature importance rankings
- decision boundaries visualization
- reasoning for each choice

### Deployment agent
what it does:
- saves model in production format
- sets up monitoring infrastructure
- creates retraining schedule
- generates deploymnet code
produces:
- saved model + preprocessors
- monitoring configuration
