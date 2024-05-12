import pandas as pd
import pulp
import matplotlib.pyplot as plt
import inspect

class ClassAssignment:
    def __init__(self, student_csv, pair_csv, classes):
        self.student_df = pd.read_csv(student_csv)
        self.pair_df = pd.read_csv(pair_csv)
        self.classes = classes
        self.problem = pulp.LpProblem('ClassAssignmentProblem', pulp.LpMaximize)
        self.students = self.student_df['student_id'].tolist()
        self.student_class_pairs = [(s, c) for s in self.students for c in self.classes]
        self.assignment_vars = pulp.LpVariable.dicts('x', self.student_class_pairs, cat='Binary')
        self.total_students = len(self.students)
        self.students_per_class = self.total_students // len(self.classes)
        
    def apply_all_constraints(self):
        # メソッド名が 'set_' で始まるすべてのメソッドを取得して実行
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("set_"):
                method()

    def set_student_class_assignment_constraints(self):
        for s in self.students:
            self.problem += pulp.lpSum([self.assignment_vars[s, c] for c in self.classes]) == 1

    def set_class_size_constraints(self):
        for c in self.classes:
            self.problem += pulp.lpSum([self.assignment_vars[s, c] for s in self.students]) <= self.students_per_class + 1

    def set_gender_balance_constraints(self):
        total_males = sum(1 for row in self.student_df.itertuples() if row.gender == 1)
        total_females = self.total_students - total_males
        males_per_class = total_males // len(self.classes)
        females_per_class = total_females // len(self.classes)
        for c in self.classes:
            self.problem += pulp.lpSum(self.assignment_vars[s, c] for s in self.students if self.student_df.loc[s-1, 'gender'] == 1) <= males_per_class + 1
            self.problem += pulp.lpSum(self.assignment_vars[s, c] for s in self.students if self.student_df.loc[s-1, 'gender'] == 0) <= females_per_class + 1
            
    def set_pair_constraints(self):
        for index, row in self.pair_df.iterrows():
            student_id1 = row['student_id1']
            student_id2 = row['student_id2']
            for c in self.classes:
                # 同一クラスに割り当てられないようにする
                self.problem += self.assignment_vars[student_id1, c] + self.assignment_vars[student_id2, c] <= 1

    def set_leader_constraints(self):
        leaders = [s for s in self.students if self.student_df.loc[s-1, 'leader_flag'] == 1]
        leaders_per_class = 1
        for c in self.classes:
            constraint = pulp.lpSum([self.assignment_vars[s, c] for s in leaders]) >= leaders_per_class
            self.problem += constraint, f"min_leaders_in_class_{c}"

    def set_support_constraints(self):
        supports = [s for s in self.students if self.student_df.loc[s-1, 'support_flag'] == 1]
        supports_per_class = len(supports) // len(self.classes) + 1
        for c in self.classes:
            constraint = pulp.lpSum([self.assignment_vars[s, c] for s in supports]) <= supports_per_class
            self.problem += constraint, f"max_supports_in_class_{c}"

    def solve(self):
        self.problem.writeLP("ClassAssignmentProblem.lp")  # これにより、問題の定義をファイルに書き出す
        status = self.problem.solve()
        if status == pulp.LpStatusInfeasible:
            self.identify_infeasibility()
        return pulp.LpStatus[status]

    def get_results(self):
        class_assignments = {}
        for c in self.classes:
            class_assignments[c] = [s for s in self.students if self.assignment_vars[s, c].value() == 1]
        return class_assignments
    
    def identify_infeasibility(self):
        # Identify which constraints are not satisfied
        print("Identifying infeasible constraints:")
        for name, constraint in self.problem.constraints.items():
            if constraint.valid():
                print(f"Constraint satisfied: {name}")
            else:
                print(f"Constraint violated: {name}")

    def plot_results(self):
        results = self.get_results()
        fig = plt.figure(figsize=(12, 20))
        for i, c in enumerate(self.classes):
            class_df = self.student_df[self.student_df['student_id'].isin(results[c])]
            ax = fig.add_subplot(4, 2, i + 1)
            ax.hist(class_df['score'], bins=range(0, 500, 40))
            ax.set(title=f'Class {c}', xlabel='Score', ylabel='Count')
        plt.tight_layout()
        plt.show()

    def save_class_assignments(self):
            # Save the assignment results to individual CSV files for each class
            results = self.get_results()
            for class_name, student_ids in results.items():
                class_df = self.student_df[self.student_df['student_id'].isin(student_ids)]
                class_df.to_csv(f'{class_name}_students.csv', index=False)
