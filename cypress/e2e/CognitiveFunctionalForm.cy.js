describe("Cognitive Functional Form", () => {
  beforeEach(() => {
    // Replace with the actual URL where the form is hosted
    cy.visit("http://localhost:3000");
  });

  it("should fill out the cognitive functional form", () => {
    // Family History of Alzheimer's - selecting 'Yes'
    cy.get("#familyHistoryAlzheimersYes").check();

    // Cardiovascular Disease - selecting 'No'
    cy.get("#cardiovascularDiseaseNo").check();

    // Diabetes - selecting 'Yes'
    cy.get("#diabetesYes").check();

    // Depression - selecting 'No'
    cy.get("#depressionNo").check();

    // Head Injury - selecting 'Yes'
    cy.get("#headInjuryYes").check();

    // Hypertension - selecting 'No'
    cy.get("#hypertensionNo").check();

    // Entering values in numeric fields
    cy.get("#systolicBP").type("120");
    cy.get("#diastolicBP").type("80");
    cy.get("#cholesterolTotal").type("200");
    cy.get("#cholesterolLDL").type("100");
    cy.get("#cholesterolHDL").type("50");
    cy.get("#cholesterolTriglycerides").type("150");

  });
});
