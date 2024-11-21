const fields = [
    {
        name: "age",
        label: "Age",
        type: "number",
        min: 0,
        required: true,
        sortOrder: 1
    },
    {
        name: "gender",
        label: "Gender",
        type: "radio",
        options: [
            { value: "0", label: "Male" },
            { value: "1", label: "Female" }
        ],
        required: true,
        sortOrder: 2
    },
    {
        name: "ethnicity",
        label: "Ethnicity",
        type: "select",
        options: [
            { value: "0", label: "Hispanic or Latino" },
            { value: "1", label: "White (Non-Hispanic)" },
            { value: "2", label: "Black or African American" },
            { value: "3", label: "Asian" }
        ],   
        required: true,
        sortOrder: 3
    },
    {
        name: "education",
        label: "Education",
        type: "select",
        options: [
            { value: "0", label: "Less than High School" },
            { value: "1", label: "High School Diploma or GED" },
            { value: "2", label: "Some College or Associate's Degree" },
            { value: "3", label: "Bachelor's Degree or Higher" }
        ],
        required: true,
        sortOrder: 4
    },
    {
        name: "bmi",
        label: "BMI",
        type: "number",
        min: 15,
        max: 50,
        step: 0.1,
        required: true,
        sortOrder: 5
    },
    {
        name: "smoking",
        label: "Smoking",
        type: "radio",
        options: [
            { value: "0", label: "No" },
            { value: "1", label: "Yes" }
        ],
        required: true,
        sortOrder: 6
    },
    {
        name: "alcohol",
        label: "Alcohol Consumption",
        type: "number",
        min: 0,
        max: 100,
        step: 0.1,
        required: true,
        sortOrder: 7
    },
    {
        name: "physical",
        label: "Physical Activity",
        type: "number",
        min: 0,
        max: 10,
        step: 0.1,
        required: true,
        sortOrder: 8
    },
    {
        name: "diet",
        label: "Diet Quality",
        type: "number",
        min: 0,
        max: 100,
        step: 0.1,
        required: true,
        sortOrder: 9
    },
    {
        name: "sleep",
        label: "Sleep",
        type: "number",
        min: 0,
        max: 24,
        step: 0.1,
        required: true,
        sortOrder: 10
    },
    {
        name: "familyHistory",
        label: "Family History",
        type: "radio",
        options: [
            { value: "0", label: "No" },
            { value: "1", label: "Yes" }
        ],
        required: true,
        sortOrder: 11
    },
    {
        name: "cardio",
        label: "Cardiovascular Disease",
        type: "radio",
        options: [
            { value: "0", label: "No" },
            { value: "1", label: "Yes" }
        ],
        required: true,
        sortOrder: 12
    },
    {
        name: "diabetes",
        label: "Diabetes",
        type: "radio",
        options: [
            { value: "0", label: "No" },
            { value: "1", label: "Yes" }
        ],
        required: true,
        sortOrder: 13
    }

]

/* - {Age, Gender, Ethnicity, Education}
- {BMI (int?), smoking(bool), alcohol, physical, diet, sleep}
- {Family history(bool), cardio(bool), diabetes(bool), depression(bool), head injury(bool), hypertension(bool), SysBP, DiaBP, Chol, CholLDL, CholHDL, CholTri}
- {MMSE, FunctAssessment, Memory(bool), Behavioral(bool), ADL(float), confusion(bool), disorientation(bool), personalityChanges(bool), difficultyTasks(bool), forgetfulness(bool)}
 */