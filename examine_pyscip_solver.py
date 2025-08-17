"""
PySCIPOpt Solver Path Inspection Framework
Enhanced with CIP model reading and comprehensive tracking
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from pyscipopt import Model, Eventhdlr, Branchrule, Nodesel, Sepa
from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, SCIP_PARAMSETTING


class SolverPathInspector:
    """Main class for inspecting solver paths and decisions"""
    
    def __init__(self, model: Optional[Model] = None, reference_cip_path: Optional[str] = None):
        self.model = model or Model()
        self.reference_cip_path = reference_cip_path
        self.reference_model = None
        
        # Data structures for tracking
        self.node_history = []
        self.solution_history = []
        self.branching_decisions = []
        self.event_log = []
        self.statistics = defaultdict(int)
        
        # Event handlers
        self.event_handler = None
        self.branching_rule = None
        self.node_selector = None
        
        # Initialize reference model if provided
        if reference_cip_path:
            self.load_reference_model(reference_cip_path)
    
    def load_reference_model(self, cip_file_path: str):
        """Load reference model from CIP file for comparison"""
        if not os.path.exists(cip_file_path):
            raise FileNotFoundError(f"CIP file not found: {cip_file_path}")
        
        print(f"Loading reference model from {cip_file_path}")
        self.reference_model = Model()
        
        try:
            # Read the CIP problem file [152]
            self.reference_model.readProblem(filename=cip_file_path)
            print(f"Successfully loaded reference model")
            print(f"Reference model has {self.reference_model.getNVars()} variables")
            print(f"Reference model has {self.reference_model.getNConss()} constraints")
            
            # Store reference model information
            self.reference_info = {
                'n_vars': self.reference_model.getNVars(),
                'n_conss': self.reference_model.getNConss(),
                'obj_sense': self.reference_model.getObjectiveSense(),
                'variables': {var.name: {'lb': var.getLbOriginal(), 
                                       'ub': var.getUbOriginal(),
                                       'obj': var.getObj(),
                                       'vtype': var.vtype()} 
                            for var in self.reference_model.getVars()},
                'constraints': [cons.name for cons in self.reference_model.getConss()]
            }
            
        except Exception as e:
            print(f"Error loading CIP file: {e}")
            self.reference_model = None
            raise
    
    def create_model_from_cip(self, cip_file_path: str) -> Model:
        """Create a new model instance from CIP file"""
        model = Model()
        model.readProblem(filename=cip_file_path) # [152]
        return model
    
    def attach_event_handlers(self):
        """Attach event handlers to track solver behavior"""
        
        # Custom event handler for tracking node processing
        class PathTrackingEventHandler(Eventhdlr):
            def __init__(self, inspector):
                self.inspector = inspector
                super().__init__()
            
            def eventexec(self, event):
                """Handle various solver events"""
                event_type = event.getType()
                current_time = datetime.now().isoformat()
                
                # Track node processing events
                if event_type == SCIP_EVENTTYPE.NODEFOCUSED:
                    node = event.getNode()
                    if node is not None:
                        node_data = {
                            'timestamp': current_time,
                            'event_type': 'NODE_FOCUSED',
                            'node_number': node.getNumber(),
                            'depth': node.getDepth(),
                            'lower_bound': node.getLowerbound(),
                            'estimate': node.getEstimate()
                        }
                        self.inspector.node_history.append(node_data)
                        self.inspector.event_log.append(node_data)
                        
                elif event_type == SCIP_EVENTTYPE.NODESOLVED:
                    node = event.getNode()
                    if node is not None:
                        # Get current LP solution if available [169]
                        lp_sol = {}
                        try:
                            if self.inspector.model.getStage() >= 3:  # Solving stage
                                for var in self.inspector.model.getVars():
                                    lp_sol[var.name] = var.getLPSol()
                        except:
                            pass
                        
                        node_data = {
                            'timestamp': current_time,
                            'event_type': 'NODE_SOLVED',
                            'node_number': node.getNumber(),
                            'depth': node.getDepth(),
                            'lp_solution': lp_sol,
                            'node_type': str(node.getType())
                        }
                        self.inspector.node_history.append(node_data)
                        self.inspector.event_log.append(node_data)
                
                # Track solution finding
                elif event_type == SCIP_EVENTTYPE.BESTSOLFOUND:
                    if self.inspector.model.getNSols() > 0:
                        best_sol = self.inspector.model.getBestSol()
                        sol_data = {
                            'timestamp': current_time,
                            'event_type': 'BEST_SOLUTION_FOUND',
                            'objective_value': self.inspector.model.getSolObjVal(best_sol),
                            'solution_values': {var.name: self.inspector.model.getSolVal(best_sol, var) 
                                              for var in self.inspector.model.getVars()},
                            'gap': self.inspector.model.getGap()
                        }
                        self.inspector.solution_history.append(sol_data)
                        self.inspector.event_log.append(sol_data)
                
                self.inspector.statistics[f'event_{event_type}'] += 1
                return {'result': SCIP_RESULT.DIDNOTRUN}
        
        # Custom branching rule for tracking branching decisions
        class BranchingTracker(Branchrule):
            def __init__(self, inspector):
                self.inspector = inspector
                super().__init__()
            
            def branchexeclp(self, allowaddcons):
                """Track branching decisions"""
                current_time = datetime.now().isoformat()
                
                # Get branching candidates
                lp_cands, lp_cands_sol, lp_cands_frac, n_lp_cands = self.model.getLPBranchCands()
                
                if n_lp_cands > 0:
                    branch_data = {
                        'timestamp': current_time,
                        'event_type': 'BRANCHING_DECISION',
                        'candidates': [{'var_name': var.name, 
                                      'lp_value': lp_cands_sol[i],
                                      'fractionality': lp_cands_frac[i]}
                                     for i, var in enumerate(lp_cands[:min(10, n_lp_cands)])],
                        'n_candidates': n_lp_cands
                    }
                    self.inspector.branching_decisions.append(branch_data)
                    self.inspector.event_log.append(branch_data)
                
                # Use default branching
                return {'result': SCIP_RESULT.DIDNOTRUN}
        
        # Initialize and include handlers
        self.event_handler = PathTrackingEventHandler(self)
        self.branching_rule = BranchingTracker(self)
        
        # Include event handler for various events [22]
        self.model.includeEventhdlr(self.event_handler, "PathTracker", 
                                   "Tracks solver path and decisions")
        
        # Catch events we're interested in
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self.event_handler)
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self.event_handler)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self.event_handler)
        
        # Include branching rule [21]
        self.model.includeBranchrule(self.branching_rule, "BranchTracker",
                                    "Tracks branching decisions", priority=-1000000)
    
    def configure_for_inspection(self, detailed_logging: bool = True):
        """Configure model settings optimal for inspection"""
        # Disable some speedup features for better tracking
        if detailed_logging:
            self.model.setPresolve(SCIP_PARAMSETTING.FAST)  # Minimal presolving
            self.model.setHeuristics(SCIP_PARAMSETTING.FAST)  # Reduce heuristics
        
        # Set parameters for better tree exploration visibility
        self.model.setParam("limits/time", 3600)  # 1 hour time limit
        self.model.setParam("display/verblevel", 4)  # Detailed output
        self.model.setParam("branching/random/priority", -1000000)  # Prefer our tracker
    
    def solve_with_tracking(self):
        """Solve the model while tracking the path"""
        print("Starting solve with path tracking...")
        start_time = datetime.now()
        
        try:
            self.model.optimize()
            
            # Gather final statistics
            self.final_stats = {
                'status': self.model.getStatus(),
                'solve_time': (datetime.now() - start_time).total_seconds(),
                'n_nodes': self.model.getNNodes(),
                'n_solutions': self.model.getNSols(),
                'final_gap': self.model.getGap() if self.model.getNSols() > 0 else float('inf'),
                'final_bound': self.model.getDualbound(),
                'final_primal': self.model.getPrimalbound() if self.model.getNSols() > 0 else float('inf')
            }
            
            print(f"Solving completed in {self.final_stats['solve_time']:.2f} seconds")
            print(f"Explored {self.final_stats['n_nodes']} nodes")
            print(f"Found {self.final_stats['n_solutions']} solutions")
            
        except Exception as e:
            print(f"Error during solving: {e}")
            raise
    
    def get_branch_tree_summary(self) -> Dict:
        """Get comprehensive branch tree traversal summary"""
        if not self.node_history:
            return {'error': 'No node history recorded'}
        
        # Analyze tree structure
        nodes_by_depth = defaultdict(list)
        for node in self.node_history:
            if 'depth' in node:
                nodes_by_depth[node['depth']].append(node)
        
        tree_stats = {
            'total_nodes_processed': len(self.node_history),
            'max_depth': max(nodes_by_depth.keys()) if nodes_by_depth else 0,
            'nodes_per_depth': {depth: len(nodes) for depth, nodes in nodes_by_depth.items()},
            'branching_decisions': len(self.branching_decisions),
            'solutions_found': len(self.solution_history)
        }
        
        return tree_stats
    
    def export_tracking_data(self, filename_prefix: str = "solver_path"):
        """Export all tracking data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to JSON for human readability
        export_data = {
            'metadata': {
                'timestamp': timestamp,
                'reference_cip': self.reference_cip_path,
                'final_stats': getattr(self, 'final_stats', {}),
                'reference_info': getattr(self, 'reference_info', {})
            },
            'node_history': self.node_history,
            'solution_history': self.solution_history,
            'branching_decisions': self.branching_decisions,
            'tree_summary': self.get_branch_tree_summary(),
            'statistics': dict(self.statistics)
        }
        
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Export event log as CSV for analysis
        if self.event_log:
            df = pd.DataFrame(self.event_log)
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
        
        print(f"Exported tracking data to:")
        print(f"  JSON: {json_filename}")
        if self.event_log:
            print(f"  CSV: {csv_filename}")
        
        return json_filename, csv_filename if self.event_log else None


class SolverInterventionEngine:
    """Engine for intervening in solver decisions based on tracked data"""
    
    def __init__(self, model: Model, inspector: SolverPathInspector):
        self.model = model
        self.inspector = inspector
        self.intervention_history = []
    
    def register_variable_fixing_callback(self, condition_func, variable_bounds: Dict[str, Tuple[float, float]]):
        """Register callback to fix variables based on solver state"""
        
        class VariableFixingHandler(Eventhdlr):
            def __init__(self, engine, condition, bounds):
                self.engine = engine
                self.condition = condition
                self.bounds = bounds
                super().__init__()
            
            def eventexec(self, event):
                if event.getType() == SCIP_EVENTTYPE.NODEFOCUSED:
                    # Check if condition is met
                    if self.condition(self.engine.model, event):
                        # Apply variable bounds
                        for var_name, (lb, ub) in self.bounds.items():
                            var = self.engine.model.getVar(var_name)
                            if var is not None:
                                self.engine.model.chgVarLb(var, lb)
                                self.engine.model.chgVarUb(var, ub)
                                
                                self.engine.intervention_history.append({
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'variable_bounds',
                                    'variable': var_name,
                                    'new_bounds': (lb, ub),
                                    'node': event.getNode().getNumber() if event.getNode() else None
                                })
                
                return {'result': SCIP_RESULT.DIDNOTRUN}
        
        handler = VariableFixingHandler(self, condition_func, variable_bounds)
        self.model.includeEventhdlr(handler, "VariableFixer", "Fixes variables based on conditions")
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, handler)


def create_solver_inspector_from_cip(cip_file_path: str, 
                                    enable_interventions: bool = False) -> Tuple[Model, SolverPathInspector, Optional[SolverInterventionEngine]]:
    """
    Factory function to create a complete solver inspection setup from CIP file
    
    Args:
        cip_file_path: Path to the CIP file containing the model
        enable_interventions: Whether to enable intervention capabilities
        
    Returns:
        Tuple of (Model, SolverPathInspector, Optional[SolverInterventionEngine])
    """
    
    # Load model from CIP file [152]
    model = Model()
    model.readProblem(filename=cip_file_path)
    
    # Create inspector with reference
    inspector = SolverPathInspector(model, cip_file_path)
    
    # Attach tracking handlers
    inspector.attach_event_handlers()
    
    # Configure for detailed inspection
    inspector.configure_for_inspection(detailed_logging=True)
    
    # Create intervention engine if requested
    intervention_engine = None
    if enable_interventions:
        intervention_engine = SolverInterventionEngine(model, inspector)
    
    print(f"Created solver inspector for model: {cip_file_path}")
    print(f"Model has {model.getNVars()} variables and {model.getNConss()} constraints")
    
    return model, inspector, intervention_engine


# Example usage and demonstration
def demo_solver_inspection(cip_file_path: str):
    """Demonstrate the solver inspection capabilities"""
    
    # Create inspection setup
    model, inspector, intervention_engine = create_solver_inspector_from_cip(
        cip_file_path, enable_interventions=True
    )
    
    # Example intervention: Fix binary variables if we're deep in the tree
    def deep_tree_condition(model, event):
        node = event.getNode()
        return node is not None and node.getDepth() > 10
    
    # Define variable bounds to apply (example)
    variable_bounds = {
        # 'x_binary_var': (0, 0),  # Fix to 0
        # 'y_binary_var': (1, 1),  # Fix to 1
    }
    
    if intervention_engine and variable_bounds:
        intervention_engine.register_variable_fixing_callback(
            deep_tree_condition, variable_bounds
        )
    
    # Solve with tracking
    inspector.solve_with_tracking()
    
    # Export results
    json_file, csv_file = inspector.export_tracking_data("demo_inspection")
    
    # Print summary
    tree_summary = inspector.get_branch_tree_summary()
    print("\n=== Solver Path Summary ===")
    for key, value in tree_summary.items():
        print(f"{key}: {value}")
    
    if hasattr(inspector, 'final_stats'):
        print("\n=== Final Statistics ===")
        for key, value in inspector.final_stats.items():
            print(f"{key}: {value}")
    
    return inspector, json_file, csv_file


if __name__ == "__main__":
    # Example: Replace with your CIP file path
    cip_path = "path/to/your/model.cip"
    
    if os.path.exists(cip_path):
        inspector, json_output, csv_output = demo_solver_inspection(cip_path)
        print(f"\nResults saved to: {json_output}")
    else:
        print(f"Please provide a valid CIP file path. Current path: {cip_path}")
        print("You can create a CIP file by saving any PySCIPOpt model:")
        print("  model.writeProblem('my_model.cip')")
