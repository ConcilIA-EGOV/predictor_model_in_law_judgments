import numpy as np
from util.parameters import FAIXAS_EXTRAVIO, FAIXAS_ATRASO, TARGET, log_file, append_to_data_log_list

current_column = ""  # Global variable to keep track of the current column being processed
first_run = False  # Global variable to indicate if it's the first run


def generate_range(value, interval_values=[]) -> int:
    for i, interval in enumerate(interval_values):
        if value < interval:
            return i
    return len(interval_values)


def hour_to_float(value, interval_values=[]) -> int:
    """
    Converte uma string de hora no formato HH:MM para um valor float representando a hora.
    """
    splits = value.split(":")
    last = len(splits) - 1
    minutes = float(splits[-last].strip()) / 60
    hours = float(splits[-(last + 1)].strip())
    f_value = hours + minutes
    f_value = generate_range(f_value, interval_values)
    global first_run
    if first_run:
        log_file.write(f" -> Coluna '{current_column}': convertendo hora '{value}' para float '{f_value}'\n")
        append_to_data_log_list("Alteracoes nas Features", f"Coluna '{current_column}': convertendo hora '{value}' para float '{f_value}'")
        first_run = False
    return f_value


def format_comma_strings(value:str, interval_values=[]) -> int | float:
    f_value = 0.0
    if 'R$' in value:
        value = value.replace('R$', '').strip()
    ttp = 'float'
    if ',' in value:
        # se forem as vírgulas dos centavos,
        # simplesmente remove-as
        tmp = value.split(',')
        if value.count(',') == 1 and (len(tmp[-1]) == 2):
            tmp = value[:-3]
        # se forem vírgulas de milhares, remove-as
        elif value.count(',') > 1 or (len(tmp[-1]) != 2):
            tmp = value.replace(',', '')
        else:
            tmp = value.replace(',', '.')
        # converte para float
        f_value = float(tmp)
    else:
        f_value = float(value)
    if interval_values:
        ttp = 'faixa'
        f_value = generate_range(f_value, interval_values)
    global first_run
    if first_run:
        log_file.write(f" -> Coluna '{current_column}': convertendo string '{value}' para {ttp} '{f_value}'\n")
        append_to_data_log_list("Alteracoes nas Features", f"Coluna '{current_column}': convertendo string '{value}' para {ttp} '{f_value}'")
        first_run = False
    return f_value


def format_binario(value, anomaly: int=0, yes: int=1, no: int=0) -> int:
    output = 0
    if type(value) == float:
        if np.isnan(value):
            output = anomaly
    elif type(value) == int:
        if value == 1:
            output = yes
        if value == 0:
            output = no
        if value == -1:
            output = anomaly
    elif type(value) == bool:
        if value:
            output = yes
        else:
            output = no
    elif type(value) == str:
        if value in ['S', 's', 'Y', 'y', 'Sim', 'sim', 'SIM', 'YES', 'Yes', 'yes', '1']:
            output = yes
        if value in ['N', 'n', 'Não', 'não', 'NÃO', 'NO', 'No', 'no', '0']:
            output = no
        if value in ['-1', '-']:
            output = anomaly
    else:
        log_file.write(f" -> Valor binário não reconhecido: {value}\n")
        output = anomaly
    global first_run
    if first_run:
        log_file.write(f" -> Coluna '{current_column}': convertendo valor binário '{value}' para '{output}'\n")
        append_to_data_log_list("Alteracoes nas Features", f"Coluna '{current_column}': convertendo valores '{type(value)}' para int")
        first_run = False
    return output
    

def format_intervalo(value, interval_values=[]):
    output = 0
    if type(value) == int:
        output =  value
    elif type(value) == float and not np.isnan(value):
        output = int(value)
    elif type(value) == str:
        f_out = 0.0
        if ',' in value:
            f_out = format_comma_strings(value, interval_values)
        elif ':' in value:
            f_out =  hour_to_float(value, interval_values)
        output = generate_range(f_out, interval_values)
    else:
        log_file.write(f" -> Valor de intervalo não reconhecido: {value}\n")
    global first_run
    if first_run:
        log_file.write(f" -> Coluna '{current_column}': convertendo valor de intervalo '{value}' para faixa '{output}'\n")
        msg = f"Coluna '{current_column}': convertendo valores '{type(value)}' para faixa"
        if interval_values:
            msg += f" com intervalos {interval_values}"
        append_to_data_log_list("Alteracoes nas Features", msg)
        first_run = False
    return output


FUNCTIONS = {
    'sentenca': lambda x: int(x),
    'ano': lambda x: int(x),
    'semestre': lambda x: int(x),
    'trimestre': lambda x: int(x),
    'direito_de_arrependimento': lambda x: format_binario(x),
    'descumprimento_de_oferta': lambda x: format_binario(x),
    'extravio_definitivo': lambda x: format_binario(x),
    'intervalo_extravio_temporario': lambda x: format_intervalo(x, FAIXAS_EXTRAVIO),
    'faixa_intervalo_extravio_temporario': lambda x: int(x),
    'violacao_furto_avaria': lambda x: format_binario(x),
    'cancelamento/alteracao_destino': lambda x: format_binario(x),
    'cancelamento': lambda x: format_binario(x),
    'intervalo_atraso': lambda x: format_intervalo(x, FAIXAS_ATRASO),
    'faixa_intervalo_atraso': lambda x: int(x),
    'culpa_exclusiva_consumidor': lambda x: format_binario(x),
    'condicoes_climaticas/fechamento_aeroporto': lambda x: format_binario(x),
    'fechamento_aeroporto': lambda x: format_binario(x),
    'noshow': lambda x: format_binario(x),
    'overbooking': lambda x: format_binario(x),
    'assistencia_cia_aerea': lambda x: format_binario(x, -1),
    'hipervulneravel': lambda x: format_binario(x),
    'dano_moral_individual': lambda x: format_comma_strings(x),
    'faixa_dano_moral_individual': lambda x: int(x),
    TARGET: lambda x: int(x),
}
