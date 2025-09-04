from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING, SCIP_LPSOLSTAT
import pandas as pd
import json
from datetime import datetime

class MyEventHandler(Eventhdlr):
    def __init__(self):
        super().__init__()
        self.last_candidates = []
        self.log = []

    def _safe_primalbound(self):
        """Return best known primal bound if available, else None"""
        sol = self.model.getBestSol()
        if sol is not None:
            return float(self.model.getSolObjVal(sol))
        return None

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        etype = event.getType()
        current_time = datetime.now().isoformat()

        if etype == SCIP_EVENTTYPE.LPSOLVED:
            if self.model.getLPSolstat() != SCIP_LPSOLSTAT.INFEASIBLE:
                try:
                    branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
                    self.last_candidates = [(v.name, float(s), float(f)) for v, s, f in zip(branch_cands, branch_cand_sols, branch_cand_fracs)]
                    entry = {
                        "event": "lp solved",
                        'timestamp': current_time,
                        "branching candidates": self.last_candidates.copy()
                    }
                    self.log.append(entry)
                except:
                    # Fallback if getLPBranchCands fails for any reason
                    self.last_candidates = []
            else:
                self.last_candidates = []

        elif etype == SCIP_EVENTTYPE.NODEBRANCHED:
            try:
                node = event.getNode()
                children_nodes = self.model.getChildren()  
                if not children_nodes:
                    print("   - No children nodes were found for this branching event.")
                    return

                # Iterate through the children to find out how they were created.
                print("**************************************************************************")
                for child_node in children_nodes:
                    parentBranchings_var, parentBranchings_val1, parentBranchings_val2 = child_node.getParentBranchings()
                    self.parentBranching = [(v.name, float(s), float(f)) for v, s, f in zip(parentBranchings_var, parentBranchings_val1, parentBranchings_val2)]
                    print(self.parentBranching)
            
                entry = {
                    "event": "branch",
                    'timestamp': current_time,
                    "node_id": node.getNumber(),
                    "depth": node.getDepth(),
                    "bound": float(node.getLowerbound()),
                    "best_sol": self._safe_primalbound(),
                    "branching_info": self.parentBranching.copy()
                }
                self.log.append(entry)
            except Exception as e:
                print(f"Error in NODE BRANCHED: {e}")
                raise

        elif etype == SCIP_EVENTTYPE.NODEFOCUSED:
            try:
                node = event.getNode()
                entry = {
                    "event": "focus",
                    'timestamp': current_time,
                    "node_id": node.getNumber(),
                    "depth": node.getDepth(),
                    "bound": float(node.getLowerbound()),
                    "best_sol": self._safe_primalbound(),
                    "candidates": None
                }
                self.log.append(entry)
            except Exception as e:
                print(f"Error in NODE FOCUSED: {e}")
                raise

        elif etype == SCIP_EVENTTYPE.BESTSOLFOUND:
            try:
                sol = self.model.getBestSol()
                obj = float(self.model.getSolObjVal(sol))
                entry = {
                    "event": "new_solution",
                    'timestamp': current_time,
                    "node_id": None,
                    "depth": None,
                    "bound": None,
                    "best_sol": obj,
                    "candidates": None
                }
                self.log.append(entry)
            except Exception as e:
                print(f"Error in BEST SOL FOUND: {e}")
                raise


def configure_model(model):
    """Configure model settings optimal for inspection"""
    # Disable some speedup features for better tracking
    model.setPresolve(SCIP_PARAMSETTING.FAST)  # Minimal presolving
    model.setHeuristics(SCIP_PARAMSETTING.FAST)  # Reduce heuristics
    
    # Set parameters for better tree exploration visibility
    model.setParam("limits/time", 3600)  # 1 hour time limit
    model.setParam("display/verblevel", 4)  # Detailed output
    model.setParam("branching/random/priority", 1000000)  # Prefer our tracker


# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":
    model = Model()
    model.readProblem("/Users/kc/Documents/AJ/project_w_kanishk/bathroom_layout_optimizer_pre_solve.cip")

    evthdlr = MyEventHandler()
    model.includeEventhdlr(evthdlr, "myevents", "Track nodes, branching, and solutions")
    configure_model(model)

    model.optimize()

    # Access structured log
    logs = evthdlr.log

    # try:
    #     df = pd.DataFrame(logs)
    #     print(df.head())
    #     df.to_csv("solver_log.csv", index=False)
    # except ImportError:
    #     print("Pandas not installed, skipping CSV export")

    with open("new_solver_log.json", 'w') as f:
        json.dump(logs, f, indent=4) # 'indent=4' makes the JSON output more readable


    print("\n--- Final Stats ---")
    if model.getBestSol() is not None:
        print(f"Best solution value: {model.getObjVal()}")
    else:
        print("No feasible solution found")
    print(f"Number of nodes: {model.getNNodes()}")
    # print(f"Number of branches: {model.getNBranchings()}") # this function does not exist
