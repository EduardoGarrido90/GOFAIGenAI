import os
import json
import re
import time
import anthropic
from anthropic import Anthropic
import subprocess
from collections import defaultdict, deque

class AdvancedClaudeToProlog:
    def __init__(self, api_key=None):
        """Initialize the ClaudeToProlog converter with Claude API key."""
        self.client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        
        self.model = "claude-3-7-sonnet-20250219"
        self.processed_topics = set()  # Track which topics have been processed
        self.fact_hash_set = set()  # Track unique facts by their hash
        self.topic_relations = defaultdict(set)  # Track relations between topics
        
    def get_related_topics(self, topic, max_topics=10):
        """Query Claude to get related topics for the main topic."""
        prompt = f"""
        I need to explore topics related to {topic}.
        
        Please list {max_topics} topics or concepts that are closely related to {topic}.
        
        Format your response as a structured JSON with the following fields:
        - "related_topics": [list of topic names]
        
        Each topic should be specific enough to be informative but general enough to yield rich information.
        Make sure your response is valid JSON without any other text.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            json_data = self._extract_json_from_response(content)
            
            # Return the list of related topics
            topics_list = json_data.get("related_topics", [])
            return topics_list
            
        except Exception as e:
            print(f"Error querying Claude for related topics: {e}")
            return []

    def get_concepts_from_claude(self, topic, domain_type="general"):
        """
        Query Claude about concepts and relationships related to a topic.
        
        Args:
            topic: The topic to query about
            domain_type: Type of domain to adapt the prompt (general, history, philosophy, literature)
        """
        # Base prompt with common elements
        base_prompt = f"""
        I need a comprehensive analysis of the topic: {topic}.
        
        Please provide:
        1. All major concepts directly related to {topic}
        2. Logical relationships between these concepts
        """
        
        # Domain-specific additions to the prompt
        domain_prompts = {
            "history": """
            3. Important temporal relationships (dates, periods, chronology)
            4. Important geographical/location relationships
            5. Key people associated with this topic and their relationships
            
            Include relationships like:
            - "born_in(person, year)"
            - "died_in(person, year)"
            - "occurred_in(event, year/period)"
            - "located_in(entity, place)"
            - "preceded(earlier_event, later_event)"
            - "succeeded(later_entity, earlier_entity)"
            - "founded_by(institution, person)"
            - "ruled_during(ruler, period)"
            - "contemporary_of(person1, person2)"
            """,
            
            "philosophy": """
            3. Temporal context of these ideas (when developed)
            4. Key philosophers/thinkers associated with this topic
            5. Schools of thought and their relationships
            
            Include relationships like:
            - "lived_during(philosopher, period)"
            - "developed_by(concept, philosopher)"
            - "influenced_by(later_philosopher, earlier_philosopher)"
            - "criticized_by(theory, critic)"
            - "response_to(later_theory, earlier_theory)"
            - "contemporary_of(philosopher1, philosopher2)"
            - "school_of_thought(philosopher, school)"
            - "main_work(philosopher, work_title)"
            """,
            
            "literature": """
            3. Temporal and geographical context
            4. Authors/creators and their relationships to the work/movement
            5. Influences and legacy
            
            Include relationships like:
            - "written_by(work, author)"
            - "published_in(work, year)"
            - "set_in(work, setting/period)"
            - "influenced_by(later_work, earlier_work)"
            - "protagonist_of(character, work)"
            - "genre_of(work, genre)"
            - "movement(work/author, literary_movement)"
            - "adapted_into(source_work, adaptation)"
            """,
            
            "arts": """
            3. Temporal and geographical context
            4. Artists/creators and their relationships to works/movements
            5. Techniques, styles, and influences
            
            Include relationships like:
            - "created_by(work, artist)"
            - "created_in(work, year/period)"
            - "belongs_to(artist/work, movement/style)"
            - "housed_in(artwork, location)"
            - "technique_used(artist/work, technique)"
            - "commissioned_by(work, patron)"
            - "influenced_by(later_artist, earlier_artist)"
            - "trained_under(artist, master)"
            """
        }
        
        # Select the appropriate domain prompt or use general
        domain_specific = domain_prompts.get(domain_type, "")
        
        # Create the final prompt
        prompt = base_prompt + domain_specific + """
        
        Format your response as a structured JSON with the following fields:
        - "concepts": [list of concept names]
        - "relationships": [list of objects containing {"source": "concept1", "relation": "relation_type", "target": "concept2", "explanation": "brief explanation"}]
        
        Make sure your response is valid JSON without any other text.
        Ensure the relation types are specific and meaningful (not just generic "related_to").
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            return self._extract_json_from_response(content)
        
        except Exception as e:
            print(f"Error querying Claude about concepts: {e}")
            raise
    
    def _extract_json_from_response(self, content):
        """Helper method to extract JSON from Claude's response."""
        # Find JSON in the response (may be wrapped in code blocks)
        json_match = re.search(r'```json\n([\s\S]*?)\n```|(\{[\s\S]*\})', content)
        if json_match:
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(2)
            return json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # As a fallback, extract anything that looks like JSON
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    try:
                        return json.loads(content[json_start:json_end])
                    except json.JSONDecodeError:
                        raise ValueError("Could not extract valid JSON from Claude's response")
                else:
                    raise ValueError("Could not find JSON in Claude's response")
    
    def generate_prolog_facts(self, data, topic):
        """Convert the JSON data from Claude into Prolog predicates."""
        # Initialize facts collection
        facts = []
        explanations = {}  # Map facts to explanations
        
        # Clean topic name for Prolog
        topic_clean = self._clean_for_prolog(topic)
        
        # Add topic as a concept
        topic_concept = f"concept({topic_clean})."
        facts.append(topic_concept)
        
        # Add concepts as facts
        for concept in data.get("concepts", []):
            # Clean concept name for Prolog
            clean_name = self._clean_for_prolog(concept)
            
            # Add concept fact
            concept_fact = f"concept({clean_name})."
            facts.append(concept_fact)
            
            # Add relationship to topic
            related_fact = f"related_to({clean_name}, {topic_clean})."
            facts.append(related_fact)
            
            # Track this relation for knowledge graph
            self.topic_relations[topic_clean].add(clean_name)
        
        # Add relationships as predicates
        for rel in data.get("relationships", []):
            source = self._clean_for_prolog(rel["source"])
            relation = self._clean_for_prolog(rel["relation"])
            target = self._clean_for_prolog(rel["target"])
            
            # Add the relationship predicate
            relationship_fact = f"{relation}({source}, {target})."
            facts.append(relationship_fact)
            
            # Store explanation
            if "explanation" in rel:
                explanation = rel["explanation"].replace("\n", " ")
                explanations[relationship_fact] = explanation
        
        return facts, explanations
    
    def _clean_for_prolog(self, text):
        """Clean a string to be a valid Prolog atom or variable."""
        if not text:
            return "unknown"
        
        # Convert to lowercase and replace spaces with underscores
        cleaned = text.lower().replace(" ", "_")
        
        # Remove special characters
        cleaned = re.sub(r'[^\w_]', '', cleaned)
        
        # Ensure it doesn't start with a capital letter (would be a variable in Prolog)
        if cleaned and cleaned[0].isupper():
            cleaned = cleaned.lower()
            
        # If it starts with a digit, prefix with 'x'
        if cleaned and cleaned[0].isdigit():
            cleaned = "x" + cleaned
            
        if not cleaned:
            return "unknown"
            
        return cleaned
    
    def create_kb_header(self):
        """Create the header for the Prolog knowledge base with comprehensive discontiguous directives"""
        # Common predicates
        common_predicates = [
            "concept/1",
            "related_to/2",
            "causes/2",
            "implies/2",
            "relates_to/2",
            "part_of/2",
            "has/2",
            "member_of/2",
            "influenced_by/2",
            "influenced/2",
            "contributed_to/2",
            "developed/2",
            "example_of/2",
            "created_by/2"
        ]
        
        # Temporal predicates
        temporal_predicates = [
            "born_in/2",
            "died_in/2",
            "occurred_in/2",
            "during/2",
            "before/2",
            "after/2",
            "began_in/2",
            "ended_in/2",
            "lasted_from/3",
            "contemporary_of/2",
            "preceded/2",
            "succeeded/2",
            "published_in/2",
            "created_in/2"
        ]
        
        # Spatial predicates
        spatial_predicates = [
            "located_in/2",
            "originated_in/2",
            "moved_to/2",
            "near/2",
            "capital_of/2",
            "region_of/2",
            "housed_in/2"
        ]
        
        # Personal predicates
        personal_predicates = [
            "founded_by/2",
            "written_by/2",
            "discovered_by/2",
            "invented_by/2",
            "ruled_by/2",
            "developed_by/2",
            "friend_of/2",
            "enemy_of/2",
            "teacher_of/2",
            "student_of/2",
            "married_to/2",
            "parent_of/2",
            "child_of/2",
            "successor_of/2",
            "predecessor_of/2",
            "colleague_of/2",
            "rival_of/2",
            "inspired/2",
            "criticized/2",
            "translated_by/2"
        ]
        
        # Domain-specific predicates
        domain_specific_predicates = [
            "school_of_thought/2",
            "main_work/2",
            "protagonist_of/2",
            "genre_of/2",
            "movement/2",
            "adapted_into/2",
            "technique_used/2",
            "commissioned_by/2",
            "trained_under/2",
            "response_to/2",
            "argued_against/2",
            "expanded_on/2",
            "similar_to/2",
            "different_from/2",
            "influenced_work/2",
            "reaction_to/2",
            "set_in/2",
            "represents/2",
            "symbolizes/2"
        ]
        
        # Combine all predicates
        all_predicates = (common_predicates + temporal_predicates + 
                         spatial_predicates + personal_predicates + 
                         domain_specific_predicates)
        
        # Create header with all discontiguous declarations
        header = [
            "% Knowledge Base generated by AdvancedClaudeToProlog",
            "% Discontiguous directives to allow predicates to be defined non-consecutively"
        ]
        
        # Add discontiguous declarations
        for predicate in all_predicates:
            header.append(f":- discontiguous {predicate}.")
        
        header.append("")  # Add an empty line at the end
        
        return header
        
    def save_to_prolog_file(self, facts, explanations, filename, overwrite=False):
        """Save Prolog facts with their explanations to a file."""
        mode = 'w' if overwrite else 'a'
        
        with open(filename, mode) as f:
            if overwrite:
                # Write header if creating a new file
                f.write("\n".join(self.create_kb_header()) + "\n")
            
            # Write facts with explanations
            for fact in facts:
                f.write(fact + "\n")
                if fact in explanations:
                    f.write(f"% Explanation: {explanations[fact]}\n")
        
        print(f"{'Created' if overwrite else 'Updated'} Prolog knowledge base in {filename}")
    
    def hash_fact(self, fact):
        """Create a unique hash for a fact to avoid duplicates"""
        # Remove whitespace and comments
        cleaned = re.sub(r'\s+', '', fact)
        return cleaned
    
    def detect_domain_type(self, topic):
        """Detect the likely domain type of a topic to tailor the query."""
        # Ask Claude to classify the topic
        prompt = f"""
        Please classify the following topic into exactly one of these domain categories:
        - history
        - philosophy
        - literature
        - arts
        - general
        
        Topic: {topic}
        
        Respond with ONLY the category name, in lowercase, no explanation needed.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            domain = response.content[0].text.strip().lower()
            
            # Validate the domain matches one of our expected categories
            valid_domains = ["history", "philosophy", "literature", "arts", "general"]
            if domain not in valid_domains:
                print(f"Unexpected domain classification: '{domain}'. Using 'general' instead.")
                domain = "general"
                
            print(f"Classified topic '{topic}' as domain: {domain}")
            return domain
            
        except Exception as e:
            print(f"Error classifying topic domain: {e}")
            return "general"  # Default to general
    
    def build_knowledge_network(self, main_topic, max_related_topics=10, max_depth=2, output_file=None):
        """
        Build a comprehensive knowledge network starting from a main topic,
        exploring to a specified depth.
        
        Args:
            main_topic: The starting topic to explore
            max_related_topics: Maximum number of related topics to explore per topic
            max_depth: How many levels deep to explore the topic network
            output_file: Output file path for the Prolog knowledge base
        """
        if output_file is None:
            topic_clean = self._clean_for_prolog(main_topic)
            output_file = f"{topic_clean}_knowledge_network.pl"
        
        # Initialize with the main topic
        self.processed_topics = set()
        self.fact_hash_set = set()
        
        # Initialize the output file with header
        with open(output_file, 'w') as f:
            f.write("\n".join(self.create_kb_header()) + "\n")
            f.write(f"% Knowledge network for main topic: {main_topic}\n")
            f.write(f"% Max depth: {max_depth}, Max related topics per node: {max_related_topics}\n\n")
        
        # BFS queue for topics: (topic, depth)
        topic_queue = deque([(main_topic, 0)])
        
        # Process topics in BFS order
        total_topics_processed = 0
        
        while topic_queue:
            current_topic, current_depth = topic_queue.popleft()
            clean_topic = current_topic.lower()
            
            # Check if already processed
            if clean_topic in self.processed_topics:
                continue
                
            # Mark as processed
            self.processed_topics.add(clean_topic)
            total_topics_processed += 1
            
            print(f"\n[{total_topics_processed}] Processing topic: {current_topic} (Depth {current_depth}/{max_depth})")
            
            try:
                # Detect the domain type for this topic
                domain_type = self.detect_domain_type(current_topic)
                
                # Get concepts and relationships for this topic
                data = self.get_concepts_from_claude(current_topic, domain_type=domain_type)
                
                # Generate Prolog facts
                facts, explanations = self.generate_prolog_facts(data, current_topic)
                
                # Filter out duplicates
                unique_facts = []
                for fact in facts:
                    fact_hash = self.hash_fact(fact)
                    if fact_hash not in self.fact_hash_set:
                        self.fact_hash_set.add(fact_hash)
                        unique_facts.append(fact)
                
                print(f"  Generated {len(unique_facts)} new unique facts about {current_topic}")
                
                # Save unique facts to the knowledge base
                self.save_to_prolog_file(unique_facts, explanations, output_file, overwrite=False)
                
                # Explore deeper if not at max depth
                if current_depth < max_depth:
                    related_topics = self.get_related_topics(current_topic, max_topics=max_related_topics)
                    print(f"  Found {len(related_topics)} related topics to explore at depth {current_depth+1}")
                    
                    # Add related topics to the queue
                    for topic in related_topics:
                        if topic.lower() not in self.processed_topics:
                            topic_queue.append((topic, current_depth + 1))
                
                # Add short delay to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing topic '{current_topic}': {e}")
        
        # Validate the Prolog file
        is_valid = self.validate_prolog_file(output_file)
        if not is_valid:
            print("\nWarning: The generated Prolog file may contain syntax errors.")
        
        print(f"\nKnowledge network built successfully in {output_file}")
        print(f"Processed {total_topics_processed} topics with {len(self.fact_hash_set)} unique facts")
        return output_file
    
    def validate_prolog_file(self, prolog_file):
        """Validate that the Prolog file is syntactically correct."""
        try:
            result = subprocess.run(
                ['swipl', '-q', '-t', 'halt', '-s', prolog_file],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Prolog validation errors: {result.stderr}")
                return False
            return True
        except subprocess.SubprocessError as e:
            print(f"Error validating Prolog file: {e}")
            return False
        except FileNotFoundError:
            print("SWI-Prolog (swipl) not found. Skipping validation.")
            return True  # Assume valid if we can't check
    
    def query_prolog(self, prolog_file, query):
        """Query the Prolog knowledge base."""
        try:
            # Run the query directly
            result = subprocess.run(
                ['swipl', '-q', '-l', prolog_file, '-g', query, '-t', 'halt'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error in Prolog query: {result.stderr}")
                
                # Try a more diagnostic approach
                print("\nAttempting diagnostic query...")
                diagnostic_query = f"consult('{prolog_file}'), writeln('File loaded successfully'), {query}."
                
                diagnostic = subprocess.run(
                    ['swipl', '-q', '-g', diagnostic_query, '-t', 'halt'],
                    capture_output=True,
                    text=True
                )
                
                print("Diagnostic output:")
                print(diagnostic.stdout)
                print(diagnostic.stderr)
                
                return "Query error. See diagnostic output."
            
            return result.stdout
            
        except subprocess.SubprocessError as e:
            print(f"Error executing Prolog query: {e}")
            return None
        except FileNotFoundError:
            print("SWI-Prolog (swipl) not found. Please install it to query the knowledge base.")
            return None
    
    def extract_all_prolog_facts(self, prolog_file):
        """Extract all facts from a Prolog file for reporting"""
        try:
            with open(prolog_file, 'r') as f:
                content = f.read()
                
            # Remove comments and directives
            content = re.sub(r'%.*', '', content)
            content = re.sub(r':- .*\.', '', content)
            
            # Extract facts (lines ending with a period)
            facts = re.findall(r'[^.%]+\.', content)
            
            # Clean facts
            cleaned_facts = [fact.strip() for fact in facts if fact.strip()]
            return cleaned_facts
            
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []
    
    def generate_knowledge_report(self, prolog_file, main_topic):
        """
        Generate a comprehensive report based on the knowledge in the Prolog file
        
        Args:
            prolog_file: Path to the Prolog knowledge base
            main_topic: The main topic of the report
        
        Returns:
            The generated report text
        """
        print(f"Generating knowledge report for {main_topic}...")
        
        # Extract facts from the Prolog file
        facts = self.extract_all_prolog_facts(prolog_file)
        
        if not facts:
            print("No facts found in the Prolog file.")
            return "No knowledge available to generate a report."
        
        # Clean the topic name for display
        display_topic = main_topic.replace('_', ' ').title()
        
        # Structure facts by predicate type
        concepts = []
        relationships = defaultdict(list)
        
        # Track temporal, spatial, and personal relationships specifically
        temporal_relations = []
        spatial_relations = []
        personal_relations = []
        
        # Pattern for extracting predicate and arguments
        fact_pattern = re.compile(r'([a-z_]+)\(([^,]+),\s*([^)]+)\)')
        
        # Temporal predicates
        temporal_predicates = {"born_in", "died_in", "occurred_in", "during", "before", "after", 
                               "began_in", "ended_in", "contemporary_of", "preceded", "succeeded",
                               "published_in", "created_in"}
        
        # Spatial predicates
        spatial_predicates = {"located_in", "originated_in", "moved_to", "near", "capital_of", 
                              "region_of", "housed_in"}
        
        # Personal predicates
        personal_predicates = {"founded_by", "written_by", "discovered_by", "invented_by", 
                               "ruled_by", "developed_by", "friend_of", "teacher_of", "student_of",
                               "married_to", "parent_of", "child_of", "successor_of", "predecessor_of"}
        
        # Process each fact
        for fact in facts:
            fact = fact.strip()
            if fact.startswith("concept("):
                # Extract concept name
                match = re.search(r'concept\(([^)]+)\)', fact)
                if match:
                    concept_name = match.group(1)
                    concepts.append(concept_name)
            else:
                # Extract relationship
                match = fact_pattern.search(fact)
                if match:
                    relation_type = match.group(1)
                    source = match.group(2)
                    target = match.group(3)
                    
                    # Store the relation
                    relationships[relation_type].append((source, target))
                    
                    # Classify the relation
                    if relation_type in temporal_predicates:
                        temporal_relations.append((relation_type, source, target))
                    elif relation_type in spatial_predicates:
                        spatial_relations.append((relation_type, source, target))
                    elif relation_type in personal_predicates:
                        personal_relations.append((relation_type, source, target))
        
        # Create a prompt for Claude to generate the report
        prompt = f"""
        I have a knowledge base about {display_topic} with the following facts:
        
        1. Concepts:
        {', '.join(concepts)}
        
        2. Relationships:
        """
        
        # Add regular relationships
        for relation_type, relation_pairs in relationships.items():
            prompt += f"\n{relation_type}:\n"
            for source, target in relation_pairs:
                source_display = source.replace('_', ' ')
                target_display = target.replace('_', ' ')
                prompt += f"- {source_display} {relation_type.replace('_', ' ')} {target_display}\n"
        
        # Emphasize temporal, spatial, and personal relationships
        if temporal_relations:
            prompt += "\n3. Temporal Relationships:\n"
            for rel_type, source, target in temporal_relations:
                source_display = source.replace('_', ' ')
                target_display = target.replace('_', ' ')
                prompt += f"- {source_display} {rel_type.replace('_', ' ')} {target_display}\n"
                
        if spatial_relations:
            prompt += "\n4. Spatial/Geographical Relationships:\n"
            for rel_type, source, target in spatial_relations:
                source_display = source.replace('_', ' ')
                target_display = target.replace('_', ' ')
                prompt += f"- {source_display} {rel_type.replace('_', ' ')} {target_display}\n"
                
        if personal_relations:
            prompt += "\n5. Personal/Biographical Relationships:\n"
            for rel_type, source, target in personal_relations:
                source_display = source.replace('_', ' ')
                target_display = target.replace('_', ' ')
                prompt += f"- {source_display} {rel_type.replace('_', ' ')} {target_display}\n"
        
        prompt += f"""
        Based on this knowledge, please write a comprehensive, well-structured report about {display_topic}.
        
        The report should:
        1. Start with an introduction to {display_topic}
        2. Cover all the major concepts and their relationships
        3. Include specific sections for temporal context, geographical context, and key people/relationships if such information is available
        4. Organize information into logical sections with headings
        5. Include a conclusion summarizing the key points
        6. Be written in an academic but accessible style
        7. Be approximately 1500-2000 words
        
        Please ensure the report flows naturally while covering all the information in the knowledge base.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            report = response.content[0].text
            
            # Save the report to a file
            report_file = f"{os.path.splitext(prolog_file)[0]}_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
                
            print(f"Report generated and saved to {report_file}")
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build a comprehensive Prolog knowledge network from Claude")
    parser.add_argument("topic", help="Main topic to start the knowledge network")
    parser.add_argument("--api-key", help="Claude API key (if not set in environment)")
    parser.add_argument("--output", help="Output Prolog file")
    parser.add_argument("--max-topics", type=int, default=10, 
                        help="Maximum number of related topics to include per topic")
    parser.add_argument("--depth", type=int, default=2,
                        help="How many levels deep to explore the topic network")
    parser.add_argument("--domain", choices=["general", "history", "philosophy", "literature", "arts"],
                        default=None, help="Specify the domain type for more targeted extraction")
    parser.add_argument("--query", help="Query to run on the generated knowledge base")
    parser.add_argument("--report", action="store_true", 
                        help="Generate a comprehensive report based on the knowledge base")
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = AdvancedClaudeToProlog(api_key=args.api_key)
    
    # Build the knowledge network
    output_file = extractor.build_knowledge_network(
        args.topic, 
        max_related_topics=args.max_topics,
        max_depth=args.depth,
        output_file=args.output
    )
    
    # Generate report if requested
    if args.report:
        report = extractor.generate_knowledge_report(output_file, args.topic)
        print("\nReport Preview (first 500 characters):")
        print(report[:500] + "...")

        
    # Run query if provided
    if args.query:
        print(f"\nRunning query: {args.query}")
        result = extractor.query_prolog(output_file, args.query)
        print(f"Query result:\n{result}")
    else:
        print(f"\nKnowledge network saved to {output_file}")
        print("You can query it using SWI-Prolog, for example:")
        print(f"  swipl -l {output_file}")
        print("Example queries:")
        print(f"  listing(concept).")
        topic_clean = extractor._clean_for_prolog(args.topic)
        print(f"  findall(X, related_to(X, {topic_clean}), Xs), length(Xs, Count), write(Count), write(' related concepts found'), nl, member(X, Xs), write(X), nl, fail.")
        
        # Show examples of temporal/spatial/personal queries if appropriate
        if args.domain in ["history", "philosophy", "literature", "arts"]:
            print("\nDomain-specific query examples:")
            
            if args.domain == "history":
                print(f"  findall(X-Y, occurred_in(X, Y), Events), member(E, Events), write(E), nl, fail.")
                print(f"  findall(X-Y, located_in(X, Y), Locations), member(L, Locations), write(L), nl, fail.")
                print(f"  findall(X-Y, (born_in(X, Y); died_in(X, Y)), People), member(P, People), write(P), nl, fail.")
                
            elif args.domain == "philosophy":
                print(f"  findall(X-Y, developed_by(X, Y), Concepts), member(C, Concepts), write(C), nl, fail.")
                print(f"  findall(X-Y, lived_during(X, Y), Philosophers), member(P, Philosophers), write(P), nl, fail.")
                print(f"  findall(X-Y, influenced_by(X, Y), Influences), member(I, Influences), write(I), nl, fail.")
                
            elif args.domain == "literature":
                print(f"  findall(X-Y, written_by(X, Y), Works), member(W, Works), write(W), nl, fail.")
                print(f"  findall(X-Y, published_in(X, Y), Publications), member(P, Publications), write(P), nl, fail.")
                print(f"  findall(X-Y, genre_of(X, Y), Genres), member(G, Genres), write(G), nl, fail.")
                
            elif args.domain == "arts":
                print(f"  findall(X-Y, created_by(X, Y), Works), member(W, Works), write(W), nl, fail.")
                print(f"  findall(X-Y, created_in(X, Y), Periods), member(P, Periods), write(P), nl, fail.")
                print(f"  findall(X-Y, technique_used(X, Y), Techniques), member(T, Techniques), write(T), nl, fail.")
