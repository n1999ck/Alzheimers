import React from "react";

export default function CognitiveFunctionalForm({ register }) {
  return (
    <div className="section">
      <div className="row">
        <div className="row">
          <div className="col-sm-4">
            {/* MMSE */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">MMSE:</label>
              <input
                type="number"
                className="form-control"
                
                id="mmse"
                {...register("mmse", {
                  min: 0,
                  max: 30,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Number between 0 and 30
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Functional Assessment */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Functional Assessment:</label>
              <input
                type="number"
                className="form-control"
                
                maxLength={80}
                id="functionalAssessment"
                {...register("functionalAssessment", {
                  min: 0,
                  max: 10,
                  step: 0.0000001,
                  required: true,
                })}
              />
              <div className="form-text">
                Number between 0 and 10
                </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Memory Complaints */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0 ">Memory Complaints:</label>
              <div>
                <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="memoryComplaintsNo"
                  {...register("memoryComplaints", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="memoryComplaintsYes"
                  {...register("memoryComplaints", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div></div>
              
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Behavioral Problems */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Behavioral Problems:</label>
              <div> 
                <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="behavioralProblemsNo"
                  {...register("behavioralProblems", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="behavioralProblemsYes"
                  {...register("behavioralProblems", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
              </div>
             
            </div>
          </div>
          <div className="col-sm-4">
            {/* ADL */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">ADL:</label>
              <input
                type="number"
                className="form-control"
                
                id="adl"
                {...register("adl", {
                  min: 0,
                  max: 10,
                  step: 0.1,
                  required: true,
                })}
              />
              <div className="form-text">
                Number between 0 and 10
                </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Confusion */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Confusion:</label>
              <div><div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="confusionNo"
                  {...register("confusion", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="confusionYes"
                  {...register("confusion", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
              </div>
              
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            {/* Disorientation */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Disorientation:</label>
              <div><div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="disorientationNo"
                  {...register("disorientation", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="disorientationYes"
                  {...register("disorientation", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
              </div>
              
            </div>
          </div>
          <div className="col-sm-4">
            {/* Personality Changes */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Personality Changes:</label>
              <div>
                <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="personalityChangesNo"
                  {...register("personalityChanges", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="personalityChangesYes"
                  {...register("personalityChanges", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
              </div>
            </div>
          </div>
          <div className="col-sm-4">
            {/* Difficulty Completing Tasks */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">
                Difficulty Completing Tasks:
              </label>
              <div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="difficultyCompletingTasksNo"
                  {...register("difficultyCompletingTasks", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="difficultyCompletingTasksYes"
                  {...register("difficultyCompletingTasks", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
              </div>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-4">
            
            {/* Forgetfulness */}
            <div className="row mb-3 mb-sm-3">
              <label className="col-form-label mb-1 mb-sm-0">Forgetfulness:</label>
              <div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="0"
                  id="forgetfulnessNo"
                  {...register("forgetfulness", { required: true })}
                />
                <label className="form-check-label">No</label>
              </div>
              <div className="form-check form-check-inline">
                <input
                  type="radio"
                  className="form-check-input"
                  value="1"
                  id="forgetfulnessYes"
                  {...register("forgetfulness", { required: true })}
                />
                <label className="form-check-label">Yes</label>
              </div>
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
