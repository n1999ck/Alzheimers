describe("MedicalHistoryForm", () => {
    beforeEach(() => {
      // Assuming your app is running locally, replace `http://localhost:3000` with your app's URL.
      cy.visit("http://localhost:3000");
    });
  
    it("enters values into each medical history field", () => {
      // Family History of Alzheimer's - Yes
      cy.get("#familyHistoryAlzheimersYes").click();
      
      // Cardiovascular Disease - Yes
      cy.get("#cardiovascularDiseaseYes").click();
  
      // Diabetes - No
      cy.get("#diabetesNo").click();
  
      // Depression - Yes
      cy.get("#depressionYes").click();
  
      // Head Injury - No
      cy.get("#headInjuryNo").click();
  
      // Hypertension - Yes
      cy.get("#hypertensionYes").click();
  
      // Systolic BP - 120
      cy.get("#systolicBP").clear().type("120");
  
      // Diastolic BP - 80
      cy.get("#diastolicBP").clear().type("80");
  
      // Cholesterol Total - 200.5
      cy.get("#cholesterolTotal").clear().type("200.5");
  
      // Cholesterol LDL - 130.5
      cy.get("#cholesterolLDL").clear().type("130.5");
  
      // Cholesterol HDL - 50.2
      cy.get("#cholesterolHDL").clear().type("50.2");
  
      // Cholesterol Triglycerides - 150.4
      cy.get("#cholesterolTriglycerides").clear().type("150.4");
  
    });
  });
  