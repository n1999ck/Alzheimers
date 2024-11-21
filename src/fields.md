# Fields

## Sample Values

- [ x]  "patientID": "333333",
- [ x]  "age": "77",
- [ x]  "gender": "0",
- [ x]  "ethnicity": "0",
- [ x]  "educationLevel": "0",
- [ x]  "bmi": "33",
- [ x]  "smoking": "0",
- [ x]  "alcoholConsumption": "1.0",
- [ x]  "physicalActivity": "1.0",
- [ x]  "dietQuality": "6.0",
- [ x]  "sleepQuality": "6.0",
- [ x] "familyHistoryAlzheimers": "0",
- [ x]  "cardiovascularDisease": "0",
- [ x]  "diabetes": "0",
- [ x]  "depression": "0",
- [ x]  "headInjury": "0",
- [ x]  "hypertension": "0",
- [ x]  "systolicBP": "120",
- [ x]  "diastolicBP": "80",
- [ x]  "cholesterolTotal": "200.0",
- [ x]  "cholesterolLDL": "120.0",
- [ x]  "cholesterolHDL": "50.0",
- [ x]  "cholesterolTriglycerides": "150.0",
- [ x]  "mmse": "16",
- [ x]  "functionalAssessment": "5.0",
- [ x]  "memoryComplaints": "0",
- [ x]  "behavioralProblems": "0",
- [ x]  "adl": "5.0",
- [ x]  "confusion": "0",
- [ x]  "disorientation": "0",
- [ x]  "personalityChanges": "0",
- [ x]  "difficultyCompletingTasks": "0",
- [ x]  "forgetfulness": "0"

## Data types

- "patientID": int
- "age": int
- "gender": bool
- "ethnicity": int,
- "educationLevel": int,
- "bmi": float,
- "smoking": bool,
- "alcoholConsumption": float,
- "physicalActivity": float,
- "dietQuality": float,
- "sleepQuality": float,
- "familyHistoryAlzheimers": bool,
- "cardiovascularDisease": bool,
- "diabetes": bool,
- "depression": bool,
- "headInjury": bool,
- "hypertension": bool,
- "systolicBP": int,
- "diastolicBP": int,
- "cholesterolTotal": float,
- "cholesterolLDL": float,
- "cholesterolHDL": float,
- "cholesterolTriglycerides": float,
- "mmse": float,
- "functionalAssessment": float,
- "memoryComplaints": bool,
- "behavioralProblems": bool,
- "adl": float,
- "confusion": bool,
- "disorientation": bool,
- "personalityChanges": bool,
- "difficultyCompletingTasks": bool,
- "forgetfulness": bool

## Groups

- {Age, Gender, Ethnicity, Education}
- {BMI (int?), smoking(bool), alcohol, physical, diet, sleep}
- {Family history(bool), cardio(bool), diabetes(bool), depression(bool), head injury(bool), hypertension(bool), SysBP, DiaBP, Chol, CholLDL, CholHDL, CholTri}
- {MMSE, FunctAssessment, Memory(bool), Behavioral(bool), ADL(float), confusion(bool), disorientation(bool), personalityChanges(bool), difficultyTasks(bool), forgetfulness(bool)}

## Groups reordered

- {Age, Gender(Move to end or make a dropdown) Ethnicity, Education}
- {BMI (int?), alcohol, physical, diet, sleep, smoking(bool)}
- {SysBP, DiaBP, Chol, CholLDL, CholHDL, CholTri, Family history(bool), cardio(bool), diabetes(bool), depression(bool), head injury(bool), hypertension(bool),}
- {MMSE, FunctAssessment, ADL(float), Memory(bool), Behavioral(bool),  confusion(bool), disorientation(bool), personalityChanges(bool), difficultyTasks(bool), forgetfulness(bool)}
