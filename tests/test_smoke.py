def test_health(client):
    r=client.get('/health'); assert r.status_code==200; assert r.json['status']=='healthy'
def test_index(client):
    assert client.get('/').status_code==200
def test_security_headers(client):
    r=client.get('/'); assert r.headers['X-Content-Type-Options']=='nosniff'; assert 'Permissions-Policy' in r.headers
def test_chatbot_input_validation(client):
    r=client.post('/chatbot',json={'message':''}); assert r.status_code==400
def test_upload_page_render(client):
    client.post('/signup', data={'username': 'drtest', 'email': 'drtest@clinic.org', 'password': 'Password123!', 'full_name': 'Dr. Test'})
    client.post('/login', data={'username': 'drtest', 'password': 'Password123!'})
    r = client.get('/upload')
    assert r.status_code == 200
    assert b'Select Pathology Specimen Image' in r.data
    assert b'Run Neural Diagnostic' in r.data
    assert b'glass.css' in r.data
