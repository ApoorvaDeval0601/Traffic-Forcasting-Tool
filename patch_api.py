from pathlib import Path

content = open('api/main.py', encoding='utf-8').read()

new_endpoint = '''
@app.get("/road-edges")
def get_road_edges():
    import json
    p = Path("data/raw/road_edges.json")
    if not p.exists():
        return {"error": "Run the road edges script first"}
    with open(p) as f:
        return json.load(f)
'''

content = content.replace(
    'if __name__ == "__main__":',
    new_endpoint + '\nif __name__ == "__main__":'
)

open('api/main.py', 'w', encoding='utf-8').write(content)

print("Patched API")