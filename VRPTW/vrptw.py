import os
import pandas as pd
import re
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import random
from typing import Tuple, Dict, Any, List
random.seed(42)


def load_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines


def extract_vehicle_data(lines: List[str]) -> Tuple[int, int]:
    for line in lines:
        if re.match(r'\d+\s+\d+', line.strip()):
            parts = line.strip().split()
            num_vehicles = int(parts[0])
            vehicle_capacity = int(parts[1])
            return num_vehicles, vehicle_capacity
    raise ValueError(
        "Número de veículos e capacidade do veículo não encontrados no arquivo")


def extract_customers_data(lines: List[str]) -> pd.DataFrame:
    customers_data = []
    customer_pattern = re.compile(r'\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+')

    for line in lines:
        if customer_pattern.match(line.strip()):
            parts = line.strip().split()
            customer = {
                'Customer': int(parts[0]),
                'XCoord': round(float(parts[1]), 2),
                'YCoord': round(float(parts[2]), 2),
                'Demand': round(float(parts[3]), 2),
                'Release Date': round(float(parts[4]), 2),
                'Due Date': round(float(parts[5]), 2),
                'Service Time': round(float(parts[6]), 2),
            }
            customers_data.append(customer)
    df_customers = pd.DataFrame(customers_data)
    return df_customers


def classic_formulation(vehicle_data: Tuple[int, int], df_customers: pd.DataFrame) -> Dict[str, Any]:
    num_vehicles, vehicle_capacity = vehicle_data
    num_customers = df_customers.shape[0] - 1

    # Criação do modelo
    model = pyo.ConcreteModel()
    # Coordenadas e demandas dos clientes
    data = df_customers.to_dict('records')
    nodes = [i for i in range(0, num_customers + 1)]
    customers = nodes[1:]
    vehicles = [i for i in range(1, num_vehicles + 1)]
    coords = [(d['XCoord'], d['YCoord']) for d in data]
    d = {i: data[i]['Demand'] for i in nodes}  # Demand
    e = {i: data[i]['Release Date'] for i in nodes}  # Release_Date
    l = {i: data[i]['Due Date'] for i in nodes}  # Due_date
    s = {i: data[i]['Service Time'] for i in nodes}  # Service_time

    def distance(i, j):
        return ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5

    dist = {(i, j): distance(i, j) for i in nodes for j in nodes}

    # Variáveis de decisão
    model.x = pyo.Var(nodes, nodes, vehicles, within=pyo.Binary)
    model.y = pyo.Var(vehicles, within=pyo.Binary)
    model.b = pyo.Var(nodes, vehicles, within=pyo.NonNegativeReals)
    Mij = [[max(l[i] +s[i]+ dist[i, j] - e[j], 0) for j in nodes] for i in nodes]
    M = max(max(Mij[i]) for i in nodes)

    # Função objetivo: minimizar a distância total percorrida
    def obj_expression(model):
        return sum(dist[i, j] * model.x[i, j, k] for i in nodes for j in nodes for k in vehicles if i != j)

    model.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    # Restrições
    # Cada cliente deve ser visitado exatamente uma vez
    def visit_once(model, i):
        return sum(model.x[i, j, k] for j in nodes for k in vehicles if i != j) == 1

    model.visit_once = pyo.Constraint(customers, rule=visit_once)

    # Capacidade dos veículos
    def capacity(model, k):
        return sum(d[i] * model.x[i, j, k] for i in customers for j in nodes if i != j) <= vehicle_capacity

    model.capacity = pyo.Constraint(vehicles, rule=capacity)

    # Cada veículo sai do depósito exatamente uma vez
    def depot(model, k):
        return sum(model.x[0, j, k] for j in customers) <= 1

    model.depot = pyo.Constraint(vehicles, rule=depot)

    # Fluxo de entrada e saída dos veículos
    def flow(model, h, k):
        return sum(model.x[i, h, k] for i in nodes if i != h) == sum(model.x[h, j, k] for j in nodes if j != h)

    model.flow = pyo.Constraint(customers, vehicles, rule=flow)

    # Sub-tour elimination
    def subtour_elimination(model, i, j, k):
        if i != j:
            return model.b[j, k] >= model.b[i, k] + s[i] + dist[i, j] - M * (1 - model.x[i, j, k])
        else:
            return pyo.Constraint.Skip

    model.subtour_elimination = pyo.Constraint(
        customers, customers, vehicles, rule=subtour_elimination)

    # Janela de tempo Release date
    def release_date(model, i, k):
        if i != 0:
            return e[i] <= model.b[i, k]
        else:
            return pyo.Constraint.Skip

    model.release_date = pyo.Constraint(nodes, vehicles, rule=release_date)

    # Janela de tempo Due date
    def due_date(model, i, k):
        if i != 0:
            return model.b[i, k] <= l[i]
        else:
            return pyo.Constraint.Skip

    model.due_date = pyo.Constraint(nodes, vehicles, rule=due_date)

    return model

def better_classic_formulation(vehicle_data: Tuple[int, int], df_customers: pd.DataFrame) -> Dict[str, Any]:
    num_vehicles, vehicle_capacity = vehicle_data
    num_customers = df_customers.shape[0] - 1

    # Criação do modelo
    model = pyo.ConcreteModel()
    # Coordenadas e demandas dos clientes
    data = df_customers.to_dict('records')
    nodes = [i for i in range(0, num_customers + 1)]
    customers = nodes[1:]
    vehicles = [i for i in range(1, num_vehicles + 1)]
    coords = [(d['XCoord'], d['YCoord']) for d in data]
    d = {i: data[i]['Demand'] for i in nodes}  # Demand
    e = {i: data[i]['Release Date'] for i in nodes}  # Release_Date
    l = {i: data[i]['Due Date'] for i in nodes}  # Due_date
    s = {i: data[i]['Service Time'] for i in nodes}  # Service_time

    def distance(i, j):
        return ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5

    dist = {(i, j): distance(i, j) for i in nodes for j in nodes}

    # Variáveis de decisão
    model.x = pyo.Var(nodes, nodes, vehicles, within=pyo.Binary)
    model.y = pyo.Var(vehicles, within=pyo.Binary)
    model.b = pyo.Var(nodes, vehicles, within=pyo.NonNegativeReals)
    Mij = [[max(l[i] +s[i]+ dist[i, j] - e[j], 0) for j in nodes] for i in nodes]
    M = max(max(Mij[i]) for i in nodes)

    # Função objetivo: minimizar a distância total percorrida
    def obj_expression(model):
        return sum(dist[i, j] * model.x[i, j, k] for i in nodes for j in nodes for k in vehicles if i != j)

    model.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    # Restrições
    # Cada cliente deve ser visitado exatamente uma vez
    def visit_once(model, i):
        return sum(model.x[i, j, k] for j in nodes for k in vehicles if i != j) == 1

    model.visit_once = pyo.Constraint(customers, rule=visit_once)

    # Capacidade dos veículos
    def capacity(model, k):
        return sum(d[i] * model.x[i, j, k] for i in customers for j in nodes if i != j) <= vehicle_capacity*model.y[k]

    model.capacity = pyo.Constraint(vehicles, rule=capacity)

    # Cada veículo sai do depósito exatamente uma vez
    def depot(model, k):
        return sum(model.x[0, j, k] for j in customers) <= 1

    model.depot = pyo.Constraint(vehicles, rule=depot)

    # Fluxo de entrada e saída dos veículos
    def flow(model, h, k):
        return sum(model.x[i, h, k] for i in nodes if i != h) == sum(model.x[h, j, k] for j in nodes if j != h)

    model.flow = pyo.Constraint(customers, vehicles, rule=flow)

    # Sub-tour elimination
    def subtour_elimination(model, i, j, k):
        if i != j:
            return model.b[j, k] >= model.b[i, k] + s[i] + dist[i, j] - M * (1 - model.x[i, j, k])
        else:
            return pyo.Constraint.Skip

    model.subtour_elimination = pyo.Constraint(
        customers, customers, vehicles, rule=subtour_elimination)

    # Janela de tempo Release date
    def release_date(model, i, k):
        if i != 0:
            return e[i] <= model.b[i, k]
        else:
            return pyo.Constraint.Skip

    model.release_date = pyo.Constraint(nodes, vehicles, rule=release_date)

    # Janela de tempo Due date
    def due_date(model, i, k):
        if i != 0:
            return model.b[i, k] <= l[i]
        else:
            return pyo.Constraint.Skip

    model.due_date = pyo.Constraint(nodes, vehicles, rule=due_date)
    
    def link_xy(model, k):
        return sum(model.x[i, j, k] for i in nodes for j in nodes if i != j) <= (num_customers + 1) * model.y[k]

    model.link_xy = pyo.Constraint(vehicles, rule=link_xy)
        
    def min_vehicles(model):
        return sum(model.y[k] for k in vehicles) >= (sum(d[i] for i in customers) / vehicle_capacity)
    model.min_vehicles = pyo.Constraint(rule=min_vehicles)

    # Stronger depot constraint
    def strong_depot(model):
        return sum(model.x[0, j, k] for j in customers for k in vehicles) == sum(model.y[k] for k in vehicles)
    model.strong_depot = pyo.Constraint(rule=strong_depot)

    def symmetry_breaking(model, k):
        if k < len(vehicles):
            return model.y[k] >= model.y[k+1]
        return pyo.Constraint.Skip
    model.symmetry_breaking = pyo.Constraint(vehicles, rule=symmetry_breaking)
    
    return model


def adapted_formulation(vehicle_data: Tuple[int, int], df_customers: pd.DataFrame) -> Dict[str, Any]:
    num_vehicles, vehicle_capacity = vehicle_data
    num_customers = df_customers.shape[0] - 1

    model = pyo.ConcreteModel()
    data = df_customers.to_dict('records')
    nodes = [i for i in range(0, num_customers + 1)]
    customers = nodes[1:]
    vehicles = [i for i in range(1, num_vehicles + 1)]
    coords = [(d['XCoord'], d['YCoord']) for d in data]
    d = {i: data[i]['Demand'] for i in nodes}  # Demand
    e = {i: data[i]['Release Date'] for i in nodes}  # Release_Date
    l = {i: data[i]['Due Date'] for i in nodes}  # Due_date
    s = {i: data[i]['Service Time'] for i in nodes}  # Service_time
    vehicles_capacities = [random.randrange(
        int(vehicle_capacity/2), int(vehicle_capacity), 10) for _ in vehicles]

    def distance(i, j):
        return ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5
    dist = {(i, j): distance(i, j) for i in nodes for j in nodes}

    Mij = [[max(l[i] +s[i]+ dist[i, j] - e[j], 0) for j in nodes] for i in nodes]
    M = max(max(Mij[i]) for i in nodes)
    model.x = pyo.Var(nodes, nodes, vehicles, within=pyo.Binary)
    model.b = pyo.Var(nodes, vehicles, within=pyo.NonNegativeReals)

    # Função objetivo: minimizar a distância total percorrida
    def obj_expression(model):
        return sum(dist[i, j] * model.x[i, j, k] for i in nodes for j in nodes for k in vehicles if i != j)

    model.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)

    # Cada cliente deve ser visitado exatamente uma vez
    def visit_once(model, i):
        return sum(model.x[i, j, k] for j in nodes for k in vehicles if i != j) == 1

    model.visit_once = pyo.Constraint(customers, rule=visit_once)

    # Capacidade dos veículos
    def capacity(model, k):
        return sum(d[i] * model.x[i, j, k] for i in customers for j in nodes if i != j) <= vehicles_capacities[k-1]

    model.capacity = pyo.Constraint(vehicles, rule=capacity)

    # Cada veículo sai do depósito exatamente uma vez
    def depot(model, k):
        return sum(model.x[0, j, k] for j in customers) <= 1

    model.depot = pyo.Constraint(vehicles, rule=depot)

    # Fluxo de entrada e saída dos veículos
    def flow(model, h, k):
        return sum(model.x[i, h, k] for i in nodes if i != h) == sum(model.x[h, j, k] for j in nodes if j != h)

    model.flow = pyo.Constraint(customers, vehicles, rule=flow)

    # Sub-tour elimination
    def subtour_elimination(model, i, j, k):
        if i != j:
            return model.b[j, k] >= model.b[i, k] + s[i] + dist[i, j] - M * (1 - model.x[i, j, k])
        else:
            return pyo.Constraint.Skip

    model.subtour_elimination = pyo.Constraint(
        customers, customers, vehicles, rule=subtour_elimination)

    # Janela de tempo Release date
    def release_date(model, i, k):
        if i != 0:
            return e[i] <= model.b[i, k]
        else:
            return pyo.Constraint.Skip

    model.release_date = pyo.Constraint(nodes, vehicles, rule=release_date)

    # Janela de tempo Due date
    def due_date(model, i, k):
        if i != 0:
            return model.b[i, k] <= l[i]
        else:
            return pyo.Constraint.Skip

    model.due_date = pyo.Constraint(nodes, vehicles, rule=due_date)

    return model


def make_routes(df_customers: pd.DataFrame, vehicle_data: Tuple[int, int], model: pyo.ConcreteModel):
    num_vehicles, _ = vehicle_data
    num_customers = df_customers.shape[0] - 1
    nodes = [i for i in range(0, num_customers + 1)]
    vehicles = [i for i in range(1, num_vehicles + 1)]

    routes = []
    used_vehicles = set()

    if pyo.value(model.obj) is not None:
        for k in vehicles:
            for i in nodes:
                if i != 0 and pyo.value(model.x[0, i, k]) > 0.5:
                    used_vehicles.add(k)
                    route = [0, i]
                    while i != 0:
                        j = i
                        for h in nodes:
                            if j != h and pyo.value(model.x[j, h, k]) > 0.5:
                                route.append(h)
                                i = h
                                break
                    routes.append(route)
    return routes, used_vehicles


def solve_model(model, log_file, solver_name='gurobi'):
    solver = pyo.SolverFactory(solver_name)
    options = {
        'Cuts': 0,
        'Presolve': 0,
        'TimeLimit': 10,
        'Threads': 6,
    }

    results = solver.solve(
        model, tee=True, logfile=log_file, options=options,)

    UB = results.problem.upper_bound if hasattr(
        results.problem, 'upper_bound') else None
    LB = results.problem.lower_bound if hasattr(
        results.problem, 'lower_bound') else None
    total_time = results.solver.time if hasattr(
        results.solver, 'time') else None

    return {
        "objective_value": round(pyo.value(model.obj), 2) if pyo.value(model.obj) is not None else None,
        "UB": UB,
        "LB": LB,
        "GAP": (UB - LB) / UB if UB and LB else None,
        "Time": total_time,
        "Num_Nodes": results.solver.iterations if hasattr(results.solver, 'iterations') else None
    }


def plot_routes(df_customers, routes, file_name):
    """Plota as rotas dos veículos."""
    plt.figure(figsize=(12, 8))

    colors = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown',
        'pink', 'gray', 'olive', 'cyan', 'navy', 'lime', 'gold', 'maroon',
        'teal', 'lavender', 'turquoise', 'tan', 'salmon', 'plum', 'orchid',
        'beige', 'mint', 'coral', 'indigo', 'violet'
    ]

    plt.scatter(df_customers['XCoord'], df_customers['YCoord'],
                c='red', marker='o', s=100, label='Clientes')


    plt.scatter(df_customers['XCoord'][0], df_customers['YCoord']
                [0], c='red', marker='s', s=200, label='Depósito')


    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            plt.plot([df_customers['XCoord'][start], df_customers['XCoord'][end]],
                     [df_customers['YCoord'][start], df_customers['YCoord'][end]], color=color, lw=2, label=f'Rota {idx + 1}' if i == 0 else "")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Rotas dos Veículos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()


def main(input_folder:str, solutions_folder:str, solver_name='gurobi'):
    num_nodes = 25

    # Listas para armazenar os resultados
    classic_results_list = []
    adapted_results_list = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_folder, file_name)
            lines = load_file(file_path)
            vehicle_data = extract_vehicle_data(lines)
            df_customers = extract_customers_data(lines)
            df_customers = df_customers.head(num_nodes + 1)


            print(f"Solving instance _{num_nodes}_{file_name} with Classic - New Model\n")
            model_classic = classic_formulation(vehicle_data, df_customers)
            log_file_classic = os.path.join(
                solutions_folder, file_name.replace('.txt', f'_{num_nodes}_new_classic.log'))
            results_classic = solve_model(
                model_classic, log_file_classic, solver_name)
            routes_classic, _ = make_routes(
                df_customers, vehicle_data, model_classic)
            plot_file_classic = os.path.join(
                solutions_folder, file_name.replace('.txt', f'_{num_nodes}_new_classic.png'))
            plot_routes(df_customers, routes_classic, plot_file_classic)

            # Adiciona os resultados à lista
            classic_results_list.append({
                'Instance': file_name,
                'Objective Value': results_classic['objective_value'],
                'UB': results_classic['UB'],
                'LB': results_classic['LB'],
                'GAP': results_classic['GAP'],
                'Time': results_classic['Time'],
                'Num_Nodes': results_classic['Num_Nodes']
            })

            # Resolução com o Adapted Model
            print(f"Solving instance _{num_nodes}_{file_name} with New Classic Model\n")
            model_adapted = better_classic_formulation(vehicle_data, df_customers)
            log_file_adapted = os.path.join(
                solutions_folder, file_name.replace('.txt', f'_{num_nodes}_adapted.log'))
            results_adapted = solve_model(
                model_adapted, log_file_adapted, solver_name)
            routes_adapted, _ = make_routes(
                df_customers, vehicle_data, model_adapted)
            plot_file_adapted = os.path.join(
                solutions_folder, file_name.replace('.txt', f'_{num_nodes}_adapted.png'))
            plot_routes(df_customers, routes_adapted, plot_file_adapted)

            adapted_results_list.append({
                'Instance': file_name,
                'Objective Value': results_adapted['objective_value'],
                'UB': results_adapted['UB'],
                'LB': results_adapted['LB'],
                'GAP': results_adapted['GAP'],
                'Time': results_adapted['Time'],
                'Num_Nodes': results_adapted['Num_Nodes']
            })


    classic_results_df = pd.DataFrame(classic_results_list)
    adapted_results_df = pd.DataFrame(adapted_results_list)


    classic_results_csv = os.path.join(
        solutions_folder, f'25-R-new-classic_results.csv')
    classic_results_df.to_csv(classic_results_csv, index=False)

    adapted_results_csv = os.path.join(
        solutions_folder, f'25-R-adapted_results.csv')
    adapted_results_df.to_csv(adapted_results_csv, index=False)


input_folder = 'Instances/R1'
solutions_folder = 'Results/'
main(input_folder, solutions_folder, 'gurobi')
