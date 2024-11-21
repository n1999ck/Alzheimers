import patientDiagnoses from './patient_diagnoses.json';

Object.keys(patientDiagnoses).forEach(key => {
    console.log("Outer level key: ", key);
    const value = patientDiagnoses[key];
    console.log(value);
    for (const key in value) {
        console.log("Inner level key: ", key);
        console.log(key, value[key]);
            
       if (value[key] === 0 || value[key] === 1) {
            if (key === "Gender" || key === "Ethnicity" || key === "EducationLevel") {
                console.log("converting" + key.toString() + " from value " + value.toString + " to number");
                value[key] = Number(value[key]);
            }
            else {
                console.log("converting" + key.toString() + " from value " + value.toString + " to string");
                value[key] === 0 ? value[key] = 0 : 1; 
            }
        }
        else {
            console.log("Converting" + key.toString() + " from value " + value.toString + " to number");
            value[key] = Number(value[key]);
        }
        
}});