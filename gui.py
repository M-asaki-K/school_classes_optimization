import streamlit as st
import os
import requests

github_token = os.getenv('GITHUB_TOKEN')
openai_key = os.getenv('OPENAI_KEY')
# GitHub リポジトリ情報
repo = 'M-asaki-K/chatbot_mathopt'
file_path = ''  # 取得したいファイルのパス

import requests
import base64

# GitHubの個人アクセストークンとリポジトリ情報
TOKEN = github_token
REPO = 'M-asaki-K/chatbot_mathopt'  # リポジトリ名
FILE_PATH = ''  # 取得したいファイルのパス

# APIエンドポイントの設定
url = f'https://api.github.com/repos/{REPO}/contents/{FILE_PATH}'
# ヘッダーにトークンを設定
headers = {'Authorization': f'token {TOKEN}'}

# リクエストの送信
response = requests.get(url, headers=headers)

# レスポンスの確認
if response.status_code == 200:
    file_content = response.json()
else:
    print('Failed to retrieve file')
    
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=openai_key)
memory = ConversationBufferMemory(memory_key="chat_history")

import requests

# GitHubからディレクトリの内容を取得
response = requests.get(url)
data = response.json()

# CSVファイルのURLを見つける
csv_urls = [item['download_url'] for item in data if item['name'].endswith('.csv')]

# CSVファイルの内容をダウンロード
csv_contents = {}
for url in csv_urls:
    filename = url.split('/')[-1]
    content = requests.get(url).text
    csv_contents[filename] = content

# CSVファイルのURLを見つける
jn_urls = [item['download_url'] for item in data if item['name'].endswith('.py')]

# CSVファイルの内容をダウンロード
jn_contents = {}
for url in jn_urls:
    filename_jn = url.split('/')[-1]
    content_jn = requests.get(url).text
    jn_contents[filename_jn] = content_jn

py_file_content = jn_contents["ClassAssignment.py"]

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# LangChainの設定
llm = ChatOpenAI(api_key=openai_key)
memory = ConversationBufferMemory(memory_key="chat_history")

# 新たな制約の入力
user_constraints = st.text_input("対話型のクラス分け最適化ツールです。デフォルトでは、各クラスの人数、成績分布、男女比、リーダー、サポート要生徒数を均一化できます。あなたが追加したい制約を入力ください。:", "例: ピアノが弾ける生徒を各クラスに一人以上入れたい。")

# プロンプトテンプレート
prompt_template = """
{py_file}を参照し、次の新たな制約「{constraints}」を反映するために追加すべきPython関数を記述せよ。コメントや不要なマークアップは含めず、関数の定義のみを提供すること。
"""

# 制約関数を生成する関数
def generate_function(py_file_content, constraints):
    prompt = PromptTemplate(input_variables=["py_file", "constraints"],
                            template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    function_code = llm_chain.run(py_file=py_file_content, constraints=constraints)
    return function_code

def clean_code(code):
    # 不要なマークダウンや言語指定を削除
    clean_code = code.replace("```python", "").replace("```", "").strip()
    return clean_code

# ボタンが押されたら関数を生成
if st.button('Generate Function'):
    function_code = clean_code(generate_function(py_file_content, user_constraints))
    st.session_state['function_code'] = function_code
    st.text_area("Generated function:", function_code, height=250)

import pandas as pd
from io import StringIO

# ユーザーがボタンを押した時にCSVフォーマットを生成
if st.button('Generate CSV Format'):
    # プロンプトテンプレートの設定
    template_script = """
    {py_file}
    上記のpyファイルを参照し、下記の新たな制約関数を反映するには、元々のデータに加えてどういうデータ形式が必要か出力せよ。出力はcsvの例（列名とはじめ数行）のみでよい。コメントやマークアップは記載しないこと。
    ＜制約関数＞
    {constraints}

    Chatbot:
    """

    # スクリプトのプロンプトテンプレート作成
    prompt_script_data = PromptTemplate(
        input_variables=["py_file", "constraints"],
        template=template_script
    )

    # 制約条件の設定
    constraints = st.session_state['function_code']

    # Pythonスクリプトの解析用LLMChainの作成と実行
    llm_chain_data = LLMChain(llm=llm, prompt=prompt_script_data)
    response_data = llm_chain_data.run(py_file=jn_contents["ClassAssignment.py"], constraints=constraints)
    # CSVフォーマットをDataFrameに変換して表示
    response_data_csv = pd.read_csv(StringIO(response_data))
    st.write(response_data_csv)

# CSVファイルアップロード
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

import shutil
import tempfile
import importlib.util
import os

def add_function_and_import(filepath, function_code, module_name):
    # 一時ファイルを作成し、新しい関数を追加
    temp_dir = tempfile.mkdtemp()
    temp_filepath = shutil.copy(filepath, temp_dir)
    with open(temp_filepath, 'a', encoding='utf-8') as file:
        file.write('\n\n' + function_code)
    
    # 確認のため、追加後のファイル内容を表示
    with open(temp_filepath, 'r', encoding='utf-8') as file:
        print(file.read())

    # 一時ファイルからモジュールを動的にインポート
    spec = importlib.util.spec_from_file_location(module_name, temp_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 一時ファイルのパスを返す
    return module, temp_filepath

def clean_up(temp_dir):
    # ディレクトリとその中身を再帰的に削除
    shutil.rmtree(temp_dir)

def escape_and_indent_python_code(code, initial_indent=4, indent_size=4):
    # インデントを追加するための関数
    def add_indent(line, num_indent):
        return ' ' * num_indent + line

    # 一行ごとに処理
    lines = code.strip().split('\n')
    indented_lines = [add_indent(line, initial_indent + (i > 0) * indent_size) for i, line in enumerate(lines)]
    return '\n'.join(indented_lines)

import tempfile

if uploaded_file is not None:
#    st.session_state['uploaded_file'] = uploaded_file  # Session stateに保存しておく
    
    # 一時ファイルを作成して内容を書き込む
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # アップロードされたファイルの内容を一時ファイルにコピー
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name  # 保存された一時ファイルのパスを取得

    function_to_add_indented = escape_and_indent_python_code(st.session_state['function_code'])
    # ファイルに関数を追加してモジュールをインポート
    class_assignment_module, temp_file = add_function_and_import('ClassAssignment.py', function_to_add_indented, 'ClassAssignment')

    # モジュールからクラスをインスタンス化して、特定の制約をテストする
    assignment = class_assignment_module.ClassAssignment(file_path, 'student_pairs.csv', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    assignment.apply_all_constraints()  # すべての制約を自動で適用
    status = assignment.solve()
    
    import matplotlib.pyplot as plt

    if status == "Optimal":
        st.success("Optimal solution found!")
        results = assignment.get_results()
        student_df = pd.read_csv(file_path)
    
        # グラフの作成
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))  # ここで4x2のグリッドを作成
        axs = axs.flatten()  # 1次元配列に変換

        for i, (class_name, student_ids) in enumerate(results.items()):
            class_df = student_df[student_df['student_id'].isin(student_ids)]
            st.write(class_df)
            
            # ヒストグラムをサブプロットに追加
            axs[i].hist(class_df['score'], bins=range(0, 500, 40), color='blue', alpha=0.7)
            axs[i].set_title(f'Class {class_name}')
            axs[i].set_xlabel('Score')
            axs[i].set_ylabel('Count')

        plt.tight_layout()  # プロット間の重なりを防ぐ
        st.pyplot(fig)  # Streamlitにプロットを表示
    else:
        st.error("Failed to find an optimal solution.")