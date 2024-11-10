import React from "react";

export default function CognitiveFunctionalForm({ register }) {
  return (
    <div className="section">
      <div className="row mb-3 mb-sm-3">
        <div className="row">
          <div className="col-sm-4">
            {/* MMSE */}
            <div className="mb-3">
              <label className="col-form-label ">MMSE:</label>
              <input
                type="number"
                className="form-control"
                placeholder="Enter a floating point number between 0 and 30."
                {...register("mmse", {
                  min: 0,
                  max: 30,
                  step: 0.1,
                  required: true,
                })}
              />
            </div>
          </div>
          <div className="col-sm-4">
            {/* Functional Assessment */}
            <div className="mb-3">
              <label className="col-form-label ">Functional Assessment:</label>
              <input
                type="number"
                className="form-control"
                placeholder="Enter a floating point number between 0 and 10."
                maxLength={80}
                {...register("functionalAssessment", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
            </div>
          </div>
          <div className="col-sm-4">
            {/* Memory Complaints */}
            <div className="mb-3">
              <label className="col-form-label ">Memory Complaints:</label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("memoryComplaints", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("memoryComplaints", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Behavioral Problems */}
            <div className="mb-3">
              <label className="col-form-label ">Behavioral Problems:</label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("behavioralProblems", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("behavioralProblems", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* ADL */}
            <div className="mb-3">
              <label className="col-form-label ">ADL:</label>
              <input
                type="number"
                className="form-control"
                placeholder="Enter a floating point number between 0 and 10."
                {...register("adl", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
            </div>
          </div>
          <div className="col-sm-4">
            {/* Confusion */}
            <div className="mb-3">
              <label className="col-form-label ">Confusion:</label>
              <div className="form-check">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("confusion", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("confusion", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Disorientation */}
            <div className="mb-3">
              <label className="col-form-label ">Disorientation:</label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("disorientation", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("disorientation", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Personality Changes */}
            <div className="mb-3">
              <label className="col-form-label ">Personality Changes:</label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("personalityChanges", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("personalityChanges", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Difficulty Completing Tasks */}
            <div className="mb-3">
              <label className="col-form-label ">
                Difficulty Completing Tasks:
              </label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("difficultyCompletingTasks", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("difficultyCompletingTasks", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            
            {/* Forgetfulness */}
            <div className="mb-3">
              <label className="col-form-label ">Forgetfulness:</label>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  {...register("forgetfulness", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  {...register("forgetfulness", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
            </div>
          </div>
          <div className="col-sm-4"></div>
          <div className="col-sm-4"></div>
        </div>
      </div>
    </div>
  );
}
