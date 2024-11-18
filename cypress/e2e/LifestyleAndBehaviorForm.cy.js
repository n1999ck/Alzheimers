describe("Lifestyle and Behavior Form", () => {
    beforeEach(() => {
      // Replace with the actual URL where the form is hosted
      cy.visit("http://localhost:3000");
    });
  
    it("should fill out the lifestyle and behavior form", () => {
      cy.get("#bmi").type("25");

      cy.get("#smokingNo").check();

      cy.get("#alcoholConsumption").type("3");

      cy.get("#physicalActivity").type("3");

      cy.get("#dietQuality").type("8");

      cy.get("#sleepQuality").type("7");
    });
  });
  