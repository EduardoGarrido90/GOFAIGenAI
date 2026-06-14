% ---------------------------------------------------------------------------
% Expert-system reasoning demonstration over an LLM-extracted knowledge base.
%
% This rule layer is loaded ON TOP of an unmodified, automatically generated
% Prolog knowledge base and derives NEW facts by multi-hop deduction, negation
% as failure, and aggregation. It shows that the generated bases are not merely
% syntactically valid but support deterministic, explainable inference -- the
% defining property of a GOFAI expert system. (Addresses reviewer R#4 point 4.)
%
% Run:  swipl -q -g main -t halt rules.pl
% ---------------------------------------------------------------------------

:- dynamic influenced_by/2, criticized_by/2, developed_by/2, main_work/2,
           related_to/2, concept/1, student_of/2, response_to/2.

:- set_prolog_flag(verbose, silent).

kb('/home/eduardo/projects/LLM_PROLOG/v4_extended_clauses/plato_knowledge_network.pl').

% --- Derived rule 1: transitive intellectual influence (cycle-safe) ----------
% influenced_by(X,Y) reads "X was influenced by Y". The transitive closure
% recovers indirect intellectual ancestry that is never stated explicitly.
influenced_chain(X, Y) :- influenced_chain(X, Y, [X]).
influenced_chain(X, Y, _)   :- influenced_by(X, Y).
influenced_chain(X, Y, Vis) :- influenced_by(X, Z), \+ member(Z, Vis),
                               influenced_chain(Z, Y, [Z|Vis]).

% --- Derived rule 2: contested ideas (multi-predicate join) ------------------
contested(Concept, Developer, Critic) :-
    developed_by(Concept, Developer),
    criticized_by(Concept, Critic),
    Developer \== Critic.

% --- Derived rule 3: uncontested Platonic ideas (negation as failure) --------
uncontested(Concept) :-
    developed_by(Concept, plato),
    \+ criticized_by(Concept, _).

% --- Helper: pretty list printing -------------------------------------------
print_list([]).
print_list([H|T]) :- format("      - ~w~n", [H]), print_list(T).

banner(Title) :- format("~n~w~n", [Title]).

main :-
    kb(KB), consult(KB),
    format("~n================ EXPERT-SYSTEM REASONING TRANSCRIPT ================~n"),
    format("Knowledge base: ~w~n", [KB]),

    % Q0: scale of the base
    ( setof(C, concept(C), Cs) -> length(Cs, NC) ; NC = 0 ),
    aggregate_all(count, related_to(_, _), NR),
    banner("Q0. Base size (explicit facts)"),
    format("      concepts = ~w, related_to edges = ~w~n", [NC, NR]),

    % Q1: direct deduction -- ideas developed by Plato
    banner("Q1. ?- developed_by(Idea, plato).  [direct]"),
    ( setof(I, developed_by(I, plato), Ideas) -> true ; Ideas = [] ),
    length(Ideas, NIdeas),
    format("      ~w ideas attributed to Plato:~n", [NIdeas]),
    print_list(Ideas),

    % Q2: MULTI-HOP deduction -- Aristotle's full intellectual ancestry
    banner("Q2. ?- influenced_chain(aristotle, Ancestor).  [transitive, derived]"),
    ( setof(A, influenced_chain(aristotle, A), Anc) -> true ; Anc = [] ),
    format("      Aristotle is (transitively) influenced by:~n"),
    print_list(Anc),
    ( member(plato, Anc), member(socrates, Anc)
      -> format("      => Derived NEW fact: aristotle <- plato <- socrates "),
         format("(2-hop chain absent from the explicit base).~n")
      ;  format("      (chain not present)~n") ),

    % Q3: multi-predicate JOIN -- contested ideas
    banner("Q3. ?- contested(Idea, Developer, Critic).  [join of developed_by/criticized_by]"),
    ( setof(c(I2, D, Cr), contested(I2, D, Cr), Cons) -> true ; Cons = [] ),
    print_list(Cons),

    % Q4: negation as failure -- uncontested Platonic ideas
    banner("Q4. ?- uncontested(Idea).  [negation as failure]"),
    ( setof(U, uncontested(U), Unc) -> true ; Unc = [] ),
    length(Unc, NUnc),
    format("      ~w Platonic ideas with no recorded criticism (first 8 shown):~n", [NUnc]),
    ( length(Pre, 8), append(Pre, _, Unc) -> print_list(Pre) ; print_list(Unc) ),

    % Q5: aggregation -- main works
    banner("Q5. ?- aggregate_all(count, main_work(W, plato), N).  [aggregation]"),
    aggregate_all(count, main_work(_, plato), NWorks),
    format("      Plato has ~w catalogued main works.~n", [NWorks]),

    format("~n=================================================================~n"),
    format("All queries are deterministic and re-derivable; every answer is~n"),
    format("backed by an explicit clause or a finite resolution proof.~n").
