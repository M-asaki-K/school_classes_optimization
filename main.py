from ClassAssignment import ClassAssignment

if __name__ == "__main__":
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # クラスリスト
    assignment = ClassAssignment('students.csv', 'student_pairs.csv', classes)
    assignment.apply_all_constraints()  # すべての制約を自動で適用

    status = assignment.solve()
    assignment.save_class_assignments()

    print(f'Status: {status}')
    assignment.plot_results()