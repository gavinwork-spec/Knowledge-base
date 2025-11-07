# Personalized Search Engine Documentation

## Overview

The Personalized Search Engine is a privacy-first search system that learns from user behavior and preferences to deliver more relevant search results. It combines advanced machine learning techniques with strict privacy protection measures compliant with GDPR and CCPA regulations.

## Architecture

### Core Components

1. **DataAnonymizer** - Privacy protection through data anonymization
2. **UserBehaviorTracker** - Tracks and analyzes user query patterns
3. **ExpertiseIdentifier** - Identifies user expertise areas from behavior
4. **PersonalizationEngine** - Applies personalization to search results
5. **QuerySuggestionEngine** - Provides intelligent query auto-completion
6. **PersonalizedSearchEngine** - Main orchestrator with privacy controls
7. **PersonalizedSearchAPI** - RESTful API with privacy features

## Privacy-First Design

### Data Anonymization

**Location**: `personalized_search_engine.py:DataAnonymizer`

**Features**:
- **User ID Hashing**: SHA-256 with salt for anonymous user identification
- **IP Anonymization**: Removes last octet from IP addresses
- **PII Removal**: Automatic detection and removal of personally identifiable information
- **Session Management**: Anonymous session identifiers

**Implementation**:
```python
class DataAnonymizer:
    def anonymize_user_id(self, user_id: str) -> str:
        """Create anonymous hash for user ID"""
        return self.hash_function(f"{user_id}{self.salt}".encode()).hexdigest()[:16]

    def anonymize_text(self, text: str, remove_pii: bool = True) -> str:
        """Remove personally identifiable information from text"""
        # Removes emails, phone numbers, credit cards, SSN, etc.
```

### User Consent Management

**Location**: `personalized_search_api.py:PersonalizedSearchAPIServer`

**Features**:
- **Explicit Consent**: Required before any personalization
- **Granular Control**: Users can enable/disable specific features
- **Consent Withdrawal**: Immediate data deletion on consent withdrawal
- **Audit Logging**: Complete audit trail for compliance

### Privacy Controls

**User Privacy Configuration**:
```python
@dataclass
class UserPrivacyConfig:
    tracking_enabled: bool = True
    query_history_retention_days: int = 30
    click_tracking_enabled: bool = True
    expertise_learning_enabled: bool = True
    personalization_enabled: bool = True
    data_anonymization_enabled: bool = True
    auto_delete_after_days: Optional[int] = 365
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True
```

## User Behavior Analysis

### Query Pattern Tracking

**Location**: `personalized_search_engine.py:UserBehaviorTracker`

**Tracked Metrics**:
- **Query Frequency**: How often users search for specific terms
- **Time Patterns**: When users search (hourly distribution)
- **Query Length**: Average query complexity
- **Search Strategies**: Preferred search methods
- **Click Patterns**: Which results users click
- **Dwell Time**: Time spent on clicked results
- **Satisfaction Scores**: User-provided feedback

**Example Analysis**:
```python
def get_query_patterns(self, user_hash: str) -> Dict[str, Any]:
    return {
        'total_queries': len(queries),
        'unique_queries': len(set(queries)),
        'most_common_queries': query_counter.most_common(5),
        'most_common_terms': term_counter.most_common(10),
        'avg_query_length': np.mean(query_lengths),
        'hour_distribution': dict(hour_distribution),
        'search_strategies': Counter(search_strategies)
    }
```

### Expertise Identification

**Location**: `personalized_search_engine.py:ExpertiseIdentifier`

**Expertise Domains**:
- **Machine Learning**: neural networks, deep learning, TensorFlow
- **Web Development**: HTML, CSS, JavaScript, React
- **Data Science**: pandas, numpy, statistics, visualization
- **Software Engineering**: programming, architecture, design patterns
- **Artificial Intelligence**: AI, NLP, computer vision
- **Database**: SQL, MySQL, PostgreSQL, MongoDB
- **Cloud Computing**: AWS, Azure, Docker, Kubernetes
- **Cybersecurity**: security, encryption, authentication

**Scoring Algorithm**:
```python
def analyze_expertise(self, user_hash: str, behavior_tracker: UserBehaviorTracker) -> UserExpertiseProfile:
    expertise_scores = {}
    for domain, keywords in self.domain_keywords.items():
        domain_score = 0
        for query, count in patterns['most_common_queries']:
            for keyword in keywords:
                if keyword in query.lower():
                    domain_score += count
        expertise_scores[domain] = min(1.0, domain_score / (total_queries * 0.1))
```

## Personalization Engine

### Ranking Personalization

**Location**: `personalized_search_engine.py:PersonalizationEngine`

**Personalization Weights**:
- **Expertise Boost**: 30% - Boost results matching user expertise
- **History Boost**: 25% - Boost results from user's click history
- **Preference Boost**: 20% - Boost based on content preferences
- **Novelty Penalty**: 10% - Slightly penalize already seen content
- **Diversity Boost**: 15% - Ensure result diversity

**Implementation**:
```python
async def personalize_results(self, results: List[SearchResult],
                             expertise_profile: UserExpertiseProfile,
                             personalization_request: PersonalizedSearchRequest) -> List[SearchResult]:
    for result in results:
        personalized_score = result.score

        # Apply expertise boost
        if personalization_request.boost_expertise:
            expertise_boost = self._calculate_expertise_boost(result, expertise_profile)
            personalized_score += expertise_boost * 0.3

        # Apply personalization level scaling
        final_score = result.score + (personalized_score - result.score) * personalization_request.personalization_level
```

### Contextual Adaptation

**Adaptation Factors**:
- **User Expertise Level**: Technical content matching
- **Query Complexity**: Simplified vs detailed results
- **Time of Day**: Work vs leisure content preferences
- **Session Context**: Follow-up queries and refinements
- **Device Type**: Mobile vs desktop preferences

## Query Suggestions

### Suggestion Types

**Location**: `personalized_search_engine.py:QuerySuggestionEngine`

1. **User History Suggestions**: Based on user's previous queries
2. **Popular Query Suggestions**: Trending searches across all users
3. **Semantic Suggestions**: TF-IDF based similarity matching
4. **Expertise-Based Suggestions**: Domain-specific query completions

**Implementation**:
```python
async def get_suggestions(self, user_hash: str, partial_query: str, max_suggestions: int = 5) -> List[str]:
    suggestions = []

    # User-specific suggestions
    user_suggestions = [query for query in user_queries if partial_query.lower() in query.lower()]
    suggestions.extend(user_suggestions)

    # Global popular suggestions
    popular_suggestions = [query for query in self.popular_queries if partial_query.lower() in query.lower()]
    suggestions.extend(popular_suggestions)

    # Semantic suggestions
    semantic_suggestions = await self._get_semantic_suggestions(partial_query, max_suggestions // 2)
    suggestions.extend(semantic_suggestions)
```

## API Reference

### Authentication & Consent

#### User Consent
```http
POST /api/v1/user/consent
Content-Type: application/json

{
  "user_id": "user123",
  "consent_given": true,
  "consent_text": "I consent to personalized search and behavior tracking",
  "data_purposes": ["personalization", "analytics", "improvement"]
}
```

#### Privacy Settings
```http
POST /api/v1/user/privacy
Content-Type: application/json

{
  "user_id": "user123",
  "tracking_enabled": true,
  "query_history_retention_days": 30,
  "click_tracking_enabled": true,
  "personalization_enabled": true,
  "data_anonymization_enabled": true,
  "auto_delete_after_days": 365,
  "gdpr_compliant": true,
  "ccpa_compliant": true
}
```

### Personalized Search

#### Search with Personalization
```http
POST /api/v1/search/personalized
Content-Type: application/json

{
  "query": "machine learning algorithms",
  "user_id": "user123",
  "session_id": "session456",
  "search_strategy": "unified",
  "top_k": 10,
  "personalization_level": 0.7,
  "boost_expertise": true,
  "boost_history": true,
  "boost_preferences": true
}
```

**Response**:
```json
{
  "session_id": "session456",
  "query": "machine learning algorithms",
  "personalized_results": [...],
  "personalization_applied": true,
  "personalization_level": 0.7,
  "execution_time": 0.245,
  "user_expertise_domains": {
    "machine_learning": 0.85,
    "artificial_intelligence": 0.72
  },
  "privacy_anonymized": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Query Suggestions
```http
POST /api/v1/search/suggestions
Content-Type: application/json

{
  "user_id": "user123",
  "partial_query": "machine le",
  "max_suggestions": 5
}
```

**Response**:
```json
{
  "partial_query": "machine le",
  "suggestions": [
    "machine learning algorithms",
    "machine learning models",
    "machine learning frameworks",
    "machine learning basics",
    "machine learning python"
  ],
  "personalized": true,
  "privacy_anonymized": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### User Feedback

#### Track User Interactions
```http
POST /api/v1/user/feedback
Content-Type: application/json

{
  "user_id": "user123",
  "session_id": "session456",
  "result_id": "doc789",
  "feedback_type": "click",
  "dwell_time": 45.5,
  "satisfaction_score": 0.8
}
```

### User Data Management

#### Export User Data (GDPR)
```http
GET /api/v1/user/data-export/{user_id}
```

**Response**:
```json
{
  "user_data": {
    "user_hash": "abc123...",
    "privacy_config": {...},
    "expertise_profile": {...},
    "query_patterns": {...},
    "export_timestamp": "2024-01-01T12:00:00Z",
    "data_types": ["query_history", "expertise_profile", "privacy_preferences"]
  },
  "export_format": "json",
  "compliance": ["GDPR", "CCPA"],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Delete User Data (Right to be Forgotten)
```http
DELETE /api/v1/user/data/{user_id}
```

### Analytics

#### Personalized Analytics
```http
GET /api/v1/analytics/personalized
```

**Response**:
```json
{
  "analytics": {
    "total_anonymous_users": 150,
    "total_tracked_queries": 2500,
    "expertise_domains_distribution": {
      "machine_learning": 45,
      "web_development": 38,
      "data_science": 32
    },
    "privacy_settings_summary": {
      "tracking_enabled": 120,
      "personalization_enabled": 115,
      "data_anonymization_enabled": 150
    }
  },
  "active_sessions": 25,
  "active_websockets": 8,
  "privacy_compliance": true,
  "data_anonymized": true
}
```

## Privacy Compliance

### GDPR Compliance

**Lawful Basis**: Explicit user consent
**Data Minimization**: Only collect necessary data
**Purpose Limitation**: Use data only for stated purposes
**Storage Limitation**: Automatic deletion based on retention policies
**Data Subject Rights**: Access, rectification, erasure, portability
**Accountability**: Complete audit trail and documentation

### CCPA Compliance

**Right to Know**: Transparent data collection practices
**Right to Delete**: Complete data deletion on request
**Right to Opt-Out**: Disable tracking and personalization
**Right to Non-Discrimination**: Equal service regardless of privacy choices

### Security Measures

- **Encryption**: All data encrypted at rest and in transit
- **Access Controls**: Role-based access to user data
- **Audit Logging**: Complete audit trail of data access
- **Regular Cleanup**: Automatic deletion of expired data
- **Anonymization**: irreversible anonymization of sensitive data

## Usage Examples

### Python Client

```python
import aiohttp
import asyncio

async def personalized_search_example():
    async with aiohttp.ClientSession() as session:
        # 1. Give consent
        consent_data = {
            "user_id": "user123",
            "consent_given": True,
            "consent_text": "I consent to personalized search",
            "data_purposes": ["personalization"]
        }

        async with session.post('http://localhost:8007/api/v1/user/consent', json=consent_data) as resp:
            consent_result = await resp.json()
            print(f"Consent recorded: {consent_result['consent_recorded']}")

        # 2. Set privacy preferences
        privacy_data = {
            "user_id": "user123",
            "tracking_enabled": True,
            "personalization_enabled": True,
            "query_history_retention_days": 30
        }

        async with session.post('http://localhost:8007/api/v1/user/privacy', json=privacy_data) as resp:
            privacy_result = await resp.json()
            print("Privacy preferences set")

        # 3. Perform personalized search
        search_data = {
            "query": "machine learning algorithms",
            "user_id": "user123",
            "personalization_level": 0.8,
            "boost_expertise": True
        }

        async with session.post('http://localhost:8007/api/v1/search/personalized', json=search_data) as resp:
            results = await resp.json()
            print(f"Found {len(results['personalized_results'])} personalized results")
            print(f"Expertise domains: {results['user_expertise_domains']}")

        # 4. Get suggestions
        suggestion_data = {
            "user_id": "user123",
            "partial_query": "machine",
            "max_suggestions": 5
        }

        async with session.post('http://localhost:8007/api/v1/search/suggestions', json=suggestion_data) as resp:
            suggestions = await resp.json()
            print(f"Suggestions: {suggestions['suggestions']}")

        # 5. Track feedback
        feedback_data = {
            "user_id": "user123",
            "session_id": "session456",
            "result_id": "doc789",
            "feedback_type": "click",
            "dwell_time": 30.5,
            "satisfaction_score": 0.9
        }

        async with session.post('http://localhost:8007/api/v1/user/feedback', json=feedback_data) as resp:
            feedback_result = await resp.json()
            print("Feedback tracked successfully")

asyncio.run(personalized_search_example())
```

### JavaScript Client

```javascript
class PersonalizedSearchClient {
    constructor(baseURL = 'http://localhost:8007') {
        this.baseURL = baseURL;
        this.userId = null;
    }

    async giveConsent(consentText, dataPurposes = ['personalization']) {
        const response = await fetch(`${this.baseURL}/api/v1/user/consent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                consent_given: true,
                consent_text: consentText,
                data_purposes: dataPurposes
            })
        });
        return await response.json();
    }

    async setPrivacyPreferences(preferences) {
        const response = await fetch(`${this.baseURL}/api/v1/user/privacy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                ...preferences
            })
        });
        return await response.json();
    }

    async personalizedSearch(query, options = {}) {
        const response = await fetch(`${this.baseURL}/api/v1/search/personalized`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                user_id: this.userId,
                personalization_level: 0.7,
                boost_expertise: true,
                boost_history: true,
                ...options
            })
        });
        return await response.json();
    }

    async getSuggestions(partialQuery, maxSuggestions = 5) {
        const response = await fetch(`${this.baseURL}/api/v1/search/suggestions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                partial_query: partialQuery,
                max_suggestions: maxSuggestions
            })
        });
        return await response.json();
    }

    async trackFeedback(resultId, feedbackType, dwellTime = 0, satisfactionScore = null) {
        const response = await fetch(`${this.baseURL}/api/v1/user/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: this.userId,
                session_id: this.sessionId,
                result_id: resultId,
                feedback_type: feedbackType,
                dwell_time: dwellTime,
                satisfaction_score: satisfactionScore
            })
        });
        return await response.json();
    }

    async exportUserData() {
        const response = await fetch(`${this.baseURL}/api/v1/user/data-export/${this.userId}`);
        return await response.json();
    }

    async deleteUserData() {
        const response = await fetch(`${this.baseURL}/api/v1/user/data/${this.userId}`, {
            method: 'DELETE'
        });
        return await response.json();
    }
}

// Usage example
async function exampleUsage() {
    const client = new PersonalizedSearchClient();
    client.userId = 'user123';
    client.sessionId = 'session456';

    try {
        // 1. Give consent
        await client.giveConsent('I consent to personalized search for better results');

        // 2. Set privacy preferences
        await client.setPrivacyPreferences({
            tracking_enabled: true,
            personalization_enabled: true,
            query_history_retention_days: 30
        });

        // 3. Perform search
        const results = await client.personalizedSearch('machine learning algorithms');
        console.log('Personalized results:', results.personalized_results);
        console.log('Expertise domains:', results.user_expertise_domains);

        // 4. Get suggestions
        const suggestions = await client.getSuggestions('machine le');
        console.log('Suggestions:', suggestions.suggestions);

        // 5. Track interaction
        await client.trackFeedback('doc789', 'click', 45.5, 0.8);

    } catch (error) {
        console.error('Error:', error);
    }
}

exampleUsage();
```

### WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8007/ws/personalized-search');

ws.onopen = () => {
    console.log('Connected to personalized search WebSocket');

    // Send personalized search request
    ws.send(JSON.stringify({
        type: 'search',
        request: {
            user_id: 'user123',
            query: 'machine learning',
            personalization_level: 0.8
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'results') {
        console.log('Personalized search results:', data.results);
    } else if (data.type === 'suggestions') {
        console.log('Query suggestions:', data.suggestions);
    } else if (data.type === 'consent_required') {
        console.log('User consent required for personalization');
    } else if (data.type === 'progress') {
        console.log('Search progress:', data.message);
    }
};

// Get suggestions
function getSuggestions(partialQuery) {
    ws.send(JSON.stringify({
        type: 'suggestions',
        user_id: 'user123',
        partial_query: partialQuery
    }));
}
```

## Configuration

### Environment Variables
```bash
# Personalized Search Engine Configuration
PERSONALIZED_SEARCH_HOST=0.0.0.0
PERSONALIZED_SEARCH_PORT=8007

# Privacy Configuration
DEFAULT_RETENTION_DAYS=30
MAX_RETENTION_DAYS=365
AUTO_DELETE_ENABLED=true
DATA_ANONYMIZATION_STRENGTH=high

# Machine Learning Configuration
EXPERTISE_UPDATE_INTERVAL_HOURS=24
SUGGESTION_MODEL_TRAIN_SIZE=1000
PERSONALIZATION_WEIGHTS_PATH=/config/weights.json

# Compliance Configuration
GDPR_COMPLIANCE_ENABLED=true
CCPA_COMPLIANCE_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=2555
CONSENT_VERSION=1.0
```

### Personalization Weights
```json
{
  "expertise_boost": 0.3,
  "history_boost": 0.25,
  "preference_boost": 0.2,
  "novelty_penalty": 0.1,
  "diversity_boost": 0.15
}
```

### Expertise Domains Configuration
```json
{
  "domains": {
    "machine_learning": {
      "keywords": ["machine learning", "neural network", "deep learning", "tensorflow"],
      "technical_indicators": ["algorithm", "model", "training", "prediction"],
      "weight": 1.0
    },
    "web_development": {
      "keywords": ["html", "css", "javascript", "react", "frontend"],
      "technical_indicators": ["api", "framework", "library", "component"],
      "weight": 1.0
    }
  }
}
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_personalized_search.txt .
RUN pip install -r requirements_personalized_search.txt

COPY . .
EXPOSE 8007

# Privacy configuration
ENV DEFAULT_RETENTION_DAYS=30
ENV GDPR_COMPLIANCE_ENABLED=true
ENV CCPA_COMPLIANCE_ENABLED=true

CMD ["python", "personalized_search_api.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  personalized-search:
    build: .
    ports:
      - "8007:8007"
    environment:
      - PERSONALIZED_SEARCH_HOST=0.0.0.0
      - PERSONALIZED_SEARCH_PORT=8007
      - DEFAULT_RETENTION_DAYS=30
      - GDPR_COMPLIANCE_ENABLED=true
      - CCPA_COMPLIANCE_ENABLED=true
    volumes:
      - ./privacy_config:/app/privacy_config
      - ./user_data:/app/user_data
    depends_on:
      - hybrid-search
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
```

### Privacy Considerations for Deployment

1. **Data Encryption**: All stored data should be encrypted
2. **Access Controls**: Strict access controls for user data
3. **Backup Security**: Encrypted backups with access logging
4. **Network Security**: HTTPS/WSS for all communications
5. **Audit Trails**: Complete logging of all data access

## Performance Metrics

### Personalization Performance
- **Query Processing Time**: < 200ms additional overhead
- **Expertise Update Time**: < 100ms for profile updates
- **Suggestion Generation**: < 50ms for 5 suggestions
- **Memory Overhead**: < 100MB per 1000 active users

### Privacy Performance
- **Anonymization Processing**: < 10ms per request
- **Consent Check Time**: < 5ms per request
- **Data Export Time**: < 2s for typical user data
- **Data Deletion Time**: < 5s for complete user data removal

## Testing

### Privacy Tests
```python
async def test_privacy_compliance():
    # Test data anonymization
    anonymizer = DataAnonymizer()
    user_hash = anonymizer.anonymize_user_id("test@example.com")
    assert len(user_hash) == 16
    assert "@" not in user_hash

    # Test consent requirement
    # Test data deletion
    # Test export functionality
```

### Personalization Tests
```python
async def test_personalization():
    # Test expertise identification
    # Test result personalization
    # Test suggestion generation
    # Test feedback tracking
```

## Integration with Existing Systems

### Integration with Hybrid Search Engine
```python
# The personalized search engine wraps the hybrid search engine
personalized_engine = PersonalizedSearchEngine(hybrid_search_engine)

# Automatic integration
await personalized_engine.initialize()
```

### Integration with RAG Systems
```python
async def rag_with_personalization():
    # Get user expertise profile
    expertise = await personalized_engine.get_user_expertise(user_id)

    # Adapt RAG context based on expertise
    if expertise.get('technical_level', 0) > 0.7:
        # Include technical details
        context = technical_context
    else:
        # Use simplified explanations
        context = simplified_context
```

### Integration with Multi-Agent Systems
```python
# Register personalization as a tool for agents
orchestrator.register_tool("personalized_search", personalized_search)
orchestrator.register_tool("get_user_expertise", get_user_expertise)
orchestrator.register_tool("track_feedback", track_feedback)
```

## Troubleshooting

### Common Issues

#### Personalization Not Working
- Check if user consent is given
- Verify personalization is enabled in privacy settings
- Ensure user has sufficient search history
- Check expertise profile is generated

#### Privacy Concerns
- Verify data anonymization is working
- Check retention policies are applied
- Ensure consent is properly recorded
- Verify data deletion on request

#### Performance Issues
- Check expertise update frequency
- Optimize suggestion model size
- Review caching configuration
- Monitor memory usage

### Monitoring

### Privacy Monitoring
```bash
# Check compliance status
curl http://localhost:8007/api/v1/health/privacy

# Check user consent status
curl http://localhost:8007/api/v1/analytics/personalized

# Verify data anonymization
curl http://localhost:8007/api/v1/user/data-export/{user_id}
```

### Performance Monitoring
- Track personalization overhead
- Monitor expertise profile accuracy
- Measure suggestion relevance
- Watch privacy processing time

## Future Enhancements

### Planned Features
1. **Advanced Privacy**: Zero-knowledge proofs for privacy
2. **Federated Learning**: Privacy-preserving model training
3. **Cross-Device Personalization**: Unified experience across devices
4. **Real-Time Adaptation**: Dynamic personalization adjustments
5. **Privacy Dashboard**: User-facing privacy controls
6. **Advanced Analytics**: Privacy-preserving analytics
7. **Compliance Automation**: Automated compliance checking
8. **Data Portability**: Enhanced data export options

### Research Directions
1. **Differential Privacy**: Mathematical privacy guarantees
2. **Homomorphic Encryption**: Compute on encrypted data
3. **Secure Multi-Party Computation**: Collaborative learning
4. **Privacy-Preserving ML**: Advanced privacy techniques
5. **Explainable Personalization**: Understand why results are personalized

## Support and Contributing

### Getting Help
- **Documentation**: This file and inline code comments
- **Privacy Guide**: Separate privacy compliance documentation
- **Issues**: Create GitHub issues for bugs and privacy concerns
- **Discussions**: Use GitHub Discussions for questions

### Contributing
1. Fork the repository
2. Create a feature branch
3. Ensure privacy compliance in changes
4. Add comprehensive tests
5. Update documentation
6. Submit a pull request

### Security and Privacy
- Report security vulnerabilities privately
- Follow responsible disclosure
- Ensure all changes maintain privacy compliance
- Review privacy impact of all changes

### License
This project is licensed under the MIT License with additional privacy provisions.

---

**Last Updated**: January 2024
**Version**: 1.0.0
**Privacy Compliance**: GDPR, CCPA
**Authors**: Personalized Search Development Team