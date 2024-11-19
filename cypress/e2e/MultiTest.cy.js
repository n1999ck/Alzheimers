import patientDiagnoses from './patient_diagnoses.json';

describe("MultiTest", () => {
    beforeEach(() => {
      // Assuming your app is running locally, replace `http://localhost:3000` with your app's URL.
      cy.visit("http://localhost:3000");
    });
  
    it("enters values into each medical history field", () => {
        console.log(patientDiagnoses);

        console.log(Object.keys(patientDiagnoses));
        console.log(patientDiagnoses["1"])
        console.log("patient1: ")

        Object.keys(patientDiagnoses).forEach(key => {
         
            const value = patientDiagnoses[key];
            console.log(value);
            for (const key in value) {
                console.log(key, value[key]);
                if (typeof value[key] === 'number') {
                    cy.get(`input[name="${key}"]`).type(value[key].toString());
                    
                } else if (value[key] === 0 || value[key] === 1) {
                    cy.get(`input[name="${key}${value[key] === 0 ? 'No' : 'Yes'}"]`).click();
                    
                }
                else if (Array.isArray(value[key])) {
                    cy.get(`select[name="${key}"]`).select(value[key][0].toString());
            }
            // if (typeof value === 'number') {

            //     cy.get(`input[name="${key}"]`).type(value.toString());
            // } else if (value === 0 || value === 1) {
            //     cy.get(`input[name="${key}${value === 0 ? 'No' : 'Yes'}"]`).click();
            // } else if (Array.isArray(value)) {
            //     cy.get(`select[name="${key}"]`).select(value[0].toString());
            // }
        }
        });
    });
  });
  