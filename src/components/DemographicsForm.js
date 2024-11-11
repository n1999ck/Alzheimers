import React from "react";

export default function DemographicsForm({ register }) {
  return (
    <div className="row">
      {/* Age */}
      <div className="col-12 mb-3">
        <label className="col-form-label ">Age:</label>
        <input
          type="number"
          className="form-control"
          placeholder="Number greater than 0"
          id="age"
          {...register("age", { min: 0, required: true })}
        />
      </div>
      {/* Gender */}
      <div className="col-12 mb-3">
        <label className="col-form-label ">Gender:</label>
        <div>
          <div className="form-check form-check-inline">
            <input
              type="radio"
              className="form-check-input"
              value="0"
              id="gender"
              {...register("gender", { required: true })}
            />
            <label className="form-check-label">Male</label>
          </div>
          <div className="form-check form-check-inline">
            <input
              type="radio"
              className="form-check-input"
              value="1"
              {...register("gender", { required: true })}
            />
            <label className="form-check-label">Female</label>
          </div>
        </div>
      </div>
      {/* Ethnicity */}
      <div className="col-12 mb-3">
        <label className="col-form-label ">Ethnicity:</label>
        <select
          className="form-select"
          id="ethnicity"
          {...register("ethnicity", { required: true })}
        >
          <option value="0">Hispanic or Latino</option>
          <option value="1">White (Non-Hispanic)</option>
          <option value="2">Black or African American</option>
          <option value="3">Asian</option>
        </select>
      </div>
      {/* Education Level */}
      <div className="col-12 mb-3">
        <label className="col-form-label ">Education Level:</label>
        <select
          className="form-select"
          id="educationLevel"
          {...register("educationLevel", { required: true })}
        >
          <option value="0">Less than High School</option>
          <option value="1">High School Diploma or GED</option>
          <option value="2">Some College or Associate's Degree</option>
          <option value="3">Bachelor’s Degree or Higher</option>
        </select>
      </div>
    </div>
  );
}
