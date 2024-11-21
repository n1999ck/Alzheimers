// Define some arrays for positive and negative diagnoses
// Make sure the right order is used to correspond to indices of the arrays
// submit
describe("FillEntireForm", () => {
  beforeEach(() => {
    // Assuming your app is running locally, replace `http://localhost:3000` with your app's URL.
    cy.visit("http://localhost:3000");
  });

  it("enters values into each medical history field", () => {
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

    cy.get("#bmi").type("25");

    cy.get("#smokingNo").check();

    cy.get("#alcoholConsumption").type("3");

    cy.get("#physicalActivity").type("3");

    cy.get("#dietQuality").type("8");

    cy.get("#sleepQuality").type("7");
    cy.get("#age").type("88");

    cy.get("#genderFemale").check();

    cy.get("#ethnicity").select("1");

    cy.get("#educationLevel").select("3");

    cy.get("#mmse").type("25");
    cy.get("#functionalAssessment").type("2");
    cy.get("#adl").type("6");

    cy.get("#memoryComplaintsYes").check();

    cy.get("#behavioralProblemsYes").check();
    cy.get("#confusionYes").check();
    cy.get("#disorientationYes").check();
    cy.get("#personalityChangesYes").check();
    cy.get("#difficultyCompletingTasksYes").check();
    cy.get("#forgetfulnessYes").check();

    cy.get('form').submit();
  });
});
