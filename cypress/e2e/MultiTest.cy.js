const patientDiagnoses = require('../../fixtures/patient_diagnoses.json');

describe("FillEntireForm", () => {


    beforeEach(() => {
      // Assuming your app is running locally, replace `http://localhost:3000` with your app's URL.
      cy.visit("http://localhost:3000");
    });
  
    it("enters values into each medical history field", () => {
        console.log(patientDiagnoses);

        Object.keys(patientDiagnoses[0]).forEach(key => {
            const value = patientDiagnoses[0][key];
            if (typeof value === 'number') {
                cy.get(`input[name="${key}"]`).type(value.toString());
            } else if (value === 0 || value === 1) {
                cy.get(`input[name="${key}${value === 0 ? 'No' : 'Yes'}"]`).click();
            } else if (Array.isArray(value)) {
                cy.get(`select[name="${key}"]`).select(value[0].toString());
            }
        });
    });
  });
  