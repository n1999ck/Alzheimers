describe("Cognitive Functional Form", () => {
    beforeEach(() => {
      // Replace with the actual URL where the form is hosted
      cy.visit("http://localhost:3000");
    });
  
    it("should fill out the demographics form", () => {
        cy.get("#age").type("88");

        cy.get("#genderFemale").check();

        cy.get("#ethnicity").select("1");

        cy.get("#educationLevel").select("3");
    });
  });
  