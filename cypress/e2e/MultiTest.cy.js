import patientDiagnoses from './patient_diagnoses.json';

describe("MultiTest", () => {
    beforeEach(() => {
      // Assuming your app is running locally, replace `http://localhost:3000` with your app's URL.
      cy.visit("http://localhost:3000");
    });

    // Object.keys(patientDiagnoses).forEach(key => {
    //     console.log("Outer level key: ", key);
    //     const value = patientDiagnoses[key];
    //     console.log(value);
    //     for (const key in value) {
    //         console.log("Inner level key: ", key);
    //         console.log(key, value[key]);
                
    //        if (value[key] === 0 || value[key] === 1) {
    //             if (key === "Gender" || key === "Ethnicity" || key === "EducationLevel") {
    //                 console.log("converting" + key.toString() + " from value " + value.toString() + " to number");
    //                 value[key] = Number(value[key]);
    //             }
    //             else {
    //                 console.log("converting" + key.toString() + " from value " + value.toString() + " to string");
    //                 value[key] === 0 ? value[key] = 0 : 1; 
    //             }
    //         }
    //         else {
    //             console.log("Converting" + key.toString() + " from value " + value.toString() + " to number");
    //             value[key] = Number(value[key]);
    //         }
            
    // }});

    // console.log(patientDiagnoses);
  
    it("enters values into each medical history field", () => {
        // console.log(patientDiagnoses);

        // // console.log(Object.keys(patientDiagnoses));
        // console.log(patientDiagnoses["1"])
        // console.log("patient1: ")

        Object.keys(patientDiagnoses).forEach(key => {
         
            const value = patientDiagnoses[key];
            //console.log(value);
            for (const key in value) {
                // console.log(key, value[key]);
                // console.log(typeof value[key]);
             
                var formattedKey = key.charAt(0).toLowerCase() + key.slice(1);
                if (key === "BMI" || key === "ADL" || key === "MMSE") {
                    formattedKey = key.toLowerCase();
                }


                if (value[key] === 0 || value[key] === 1) {
                    if (key !== "Gender" && key !== "Ethnicity" && key !== "EducationLevel") {
                        cy.get(`#${formattedKey}${value[key] === 0 ? 'No' : 'Yes'}`).click();
                    }
                    else {
                        if (key === "Gender") {
                            cy.get(`#gender${value[key] === 0 ? 'Male' : 'Female'}`).click();
                        }
                        else{
                            cy.get(`#${formattedKey}`).select(value[key].toString());
                        }
                    }
                }
                else {
                    cy.get(`#${formattedKey}`).type(value[key]);
                }
                    

            
        }

        cy.get("form").submit();
        cy.get("")
        var closeButton = cy.get("button").last().click();
        });
    });
  });
  