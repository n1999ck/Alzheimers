describe("Cognitive Functional Form", () => {
  beforeEach(() => {
    // Replace with the actual URL where the form is hosted
    cy.visit("http://localhost:3000");
  });

  it("should fill out the cognitive functional form", () => {
    // Entering values in numeric fields
    cy.get("#mmse").type("25");
    cy.get("#functionalAssessment").type("20");
    cy.get("#adl").type("6");

    cy.get("#memoryComplaintsYes").check();

    cy.get("#behavioralProblemsYes").check();
    cy.get("#confusionYes").check();
    cy.get("#disorientationYes").check();
    cy.get("#personalityChangesYes").check();
    cy.get("#difficultyCompletingTasksYes").check();
    cy.get("#forgetfulnessYes").check();

  });
});
