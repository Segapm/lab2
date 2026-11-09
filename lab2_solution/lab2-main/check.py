from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations
from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput

print(analyze_titanic(TitanicInput("titanic.csv")))
print(analyze_brain_correlations(BrainDataInput("brainsize.txt")))