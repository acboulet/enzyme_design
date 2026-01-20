import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
import argparse
import numpy as np


class PrepareProtein():
    def __init__(self, input_cif="", ligand_name="IHP", distance_threshold=None, output_path=None):
        """
        Initialize PrepareProtein with a CIF file path.
        
        Parameters:
        -----------
        input_cif : str
            Path to the CIF protein structure file
        """
        self.input_cif = input_cif
        self.structure_id = None
        self.structure = None
        self.model = None

        self.protein_coordinates = None
        
        if input_cif:
            self.import_cif(input_cif)
            self.extract_ca_coordinates()
            self.find_residues_near_ligand(ligand_name=ligand_name, distance_threshold=distance_threshold)
            self.add_ligand_coordinates(ligand_name=ligand_name)
            self.export_coordinates_csv(output_path=output_path)
    
    def import_cif(self, cif_path):
        """
        Import CIF protein structure file using BioPython's MMCIFParser.
        
        Parameters:
        -----------
        cif_path : str
            Path to the CIF file
            
        Returns:
        --------
        Bio.PDB.Structure
            The parsed protein structure
        """
        parser = MMCIFParser(QUIET=True)
        # Extract structure ID from filename (e.g., "1DKP" from "1DKP.cif")
        self.structure_id = cif_path.split('/')[-1].split('.')[0].upper()
        self.structure = parser.get_structure(self.structure_id, cif_path)
        self.model = self.structure[0]
        
        print(f"✓ Structure loaded: {self.structure_id}")
        print(f"  Models: {len(self.structure)}")
        print(f"  Chains: {len(list(self.model.get_chains()))}")
        print(f"  Residues: {len(list(self.model.get_residues()))}")
        print(f"  Atoms: {len(list(self.model.get_atoms()))}")
        
        return self.structure

    def extract_ca_coordinates(self):
        """
        Extract alpha carbon (CA) coordinates for all residues and store as DataFrame.
        
        Sets self.protein_coordinates with columns:
        - chain: Chain ID
        - residue_name: 3-letter amino acid code
        - residue_number: Residue sequence number
        - residue_label: Combined residue_name_residue_number
        - x, y, z: Coordinates of alpha carbon
        
        Returns:
        --------
        pd.DataFrame : DataFrame with CA coordinates for all residues
        """
        if not self.model:
            print("No structure loaded. Call import_cif() first.")
            return pd.DataFrame()
        
        coordinates = []
        
        for chain in self.model:
            for residue in chain:
                # Only check standard amino acids
                if residue.get_id()[0] == ' ':
                    # Get CA (alpha carbon) atom
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coord = ca_atom.get_coord()
                        residue_name = residue.get_resname()
                        residue_number = residue.get_id()[1]
                        
                        coordinates.append({
                            'chain': chain.id,
                            'residue_name': residue_name,
                            'residue_number': residue_number,
                            'residue_label': f"{residue_name}_{residue_number}",
                            'x': round(coord[0], 3),
                            'y': round(coord[1], 3),
                            'z': round(coord[2], 3)
                        })
        
        # Convert to DataFrame and store
        self.protein_coordinates = pd.DataFrame(coordinates)
        print(f"✓ Extracted {len(self.protein_coordinates)} CA coordinates")
        return self.protein_coordinates

    def identify_ligands(self):
        """
        Identify ligands (heteroatoms that are not water) in the structure.
        
        Returns:
        --------
        list : List of dictionaries containing ligand information
        """
        if not self.model:
            print("No structure loaded. Call import_cif() first.")
            return []
        
        ligands = []
        
        for chain in self.model:
            for residue in self.model:
                hetero_flag = residue.get_id()[0]
                
                # 'H_' prefix indicates heteroatom (ligand, ion, etc.)
                if hetero_flag.startswith('H_'):
                    residue_name = residue.get_resname()
                    if residue_name != 'HOH':  # Exclude water molecules
                        ligands.append({
                            'chain': chain.id,
                            'residue_name': residue_name,
                            'residue_id': residue.get_id(),
                            'num_atoms': len(residue)
                        })
        
        return ligands
    
    def find_residues_near_ligand(self, ligand_name='IHP', distance_threshold=None):
        """
        Calculate distances from all amino acid residues to a ligand.
        Adds a 'min_distance' column to self.protein_coordinates.
        
        Parameters:
        -----------
        ligand_name : str
            The name of the ligand to calculate distances from (default: 'IHP')
        distance_threshold : float, optional
            Distance threshold in Angstroms for filtering (default: None - calculates for all residues)
        
        Returns:
        --------
        pd.DataFrame : Updated protein_coordinates DataFrame with min_distance column
        """
        if not self.model:
            print("No structure loaded. Call import_cif() first.")
            return pd.DataFrame()
        
        if self.protein_coordinates is None or len(self.protein_coordinates) == 0:
            print("No protein coordinates found. Call extract_ca_coordinates() first.")
            return pd.DataFrame()
        
        # Get all ligand atoms
        ligand_atoms = []
        for chain in self.model:
            for residue in chain:
                if residue.get_id()[0].startswith('H_') and residue.get_resname() == ligand_name:
                    ligand_atoms.extend(list(residue.get_atoms()))
        
        if not ligand_atoms:
            print(f"No ligand '{ligand_name}' found in structure")
            return self.protein_coordinates
        
        # Calculate distances for all residues in protein_coordinates
        distances = []

        for chain in self.model:
            for residue in chain:
                # Only check standard amino acids
                if residue.get_id()[0] == ' ':
                    # Calculate minimum distance between any ligand atom and any residue atom
                    min_distance = float('inf')
                    
                    for res_atom in residue.get_atoms():
                        res_coord = res_atom.get_coord()
                        
                        for lig_atom in ligand_atoms:
                            lig_coord = lig_atom.get_coord()
                            distance = np.linalg.norm(res_coord - lig_coord)
                            
                            if distance < min_distance:
                                min_distance = distance
                    
                    distances.append({
                        'residue_number': residue.get_id()[1],
                        'min_distance': round(min_distance, 2)
                    })
        
        # Create DataFrame from distances and merge with protein_coordinates
        distances_df = pd.DataFrame(distances)
        self.protein_coordinates = self.protein_coordinates.merge(
            distances_df,
            on='residue_number',
            how='left'
        )
        
        # Apply optional distance threshold filter
        if distance_threshold is not None:
            self.protein_coordinates = self.protein_coordinates[
                self.protein_coordinates['min_distance'] <= distance_threshold
            ].reset_index(drop=True)
        
        print(f"✓ Added min_distance to {len(self.protein_coordinates)} residues for {ligand_name}")
        return self.protein_coordinates
    
    def add_ligand_coordinates(self, ligand_name='IHP'):
        """
        Extract ligand atom coordinates and add them to protein_coordinates DataFrame.
        
        Parameters:
        -----------
        ligand_name : str
            The name of the ligand to extract (default: 'IHP')
        
        Returns:
        --------
        pd.DataFrame : Updated protein_coordinates DataFrame with ligand atoms
        """
        if not self.model:
            print("No structure loaded. Call import_cif() first.")
            return pd.DataFrame()
        
        if self.protein_coordinates is None:
            print("No protein coordinates found. Call extract_ca_coordinates() first.")
            return pd.DataFrame()
        
        # Get all ligand atoms
        ligand_coords = []
        atom_index = 0
        
        for chain in self.model:
            for residue in chain:
                if residue.get_id()[0].startswith('H_') and residue.get_resname() == ligand_name:
                    for atom in residue:
                        coord = atom.get_coord()
                        ligand_coords.append({
                            'chain': ligand_name,
                            'residue_name': ligand_name,
                            'residue_number': atom_index,
                            'residue_label': f"{ligand_name}_{atom.get_name()}",
                            'x': round(coord[0], 3),
                            'y': round(coord[1], 3),
                            'z': round(coord[2], 3),
                            'min_distance': 0.0
                        })
                        atom_index += 1
        
        if ligand_coords:
            # Create DataFrame from ligand coordinates
            ligand_df = pd.DataFrame(ligand_coords)
            # Append to protein_coordinates
            self.protein_coordinates = pd.concat(
                [self.protein_coordinates, ligand_df],
                ignore_index=True
            )
            print(f"✓ Added {len(ligand_coords)} ligand ({ligand_name}) atoms to coordinates")
        else:
            print(f"No ligand '{ligand_name}' found in structure")
        
        return self.protein_coordinates

    def export_coordinates_csv(self, output_path=None):
        """
        Export protein_coordinates DataFrame to a CSV file.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the CSV file. If None, uses input filename with .csv extension
            in the current directory (e.g., '1DKP.csv')
        
        Returns:
        --------
        str : Path where the file was saved
        """
        if self.protein_coordinates is None or len(self.protein_coordinates) == 0:
            print("No protein coordinates to export. Call extract_ca_coordinates() first.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            if self.input_cif:
                # Extract filename without extension and add .csv
                output_path = f"./output/{self.structure_id}.protein_coordinates.csv"
            else:
                output_path = f"{output_path}/{self.structure_id}.protein_coordinates.csv"
        
        # Export to CSV
        self.protein_coordinates.to_csv(output_path, index=False)
        print(f"✓ Exported {len(self.protein_coordinates)} coordinates to {output_path}")
        return output_path

       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare protein structure from CIF file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_cif",
        type=str,
        required=True,
        help="Path to input CIF file"
    )
    parser.add_argument(
        "--ligand",
        type=str,
        default="IHP",
        help="Ligand name to search for (default: IHP)"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Distance threshold in Angstroms (default: None - no filtering)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output CSV file (default: same name as input CIF with .csv extension)"
    )
    
    args = parser.parse_args()
    
    # Initialize protein preparation
    prep = PrepareProtein(input_cif=args.input_cif)
    
    # Find residues near ligand
    if prep.structure:
        print(f"Identifying residues near ligand '{args.ligand}'...")
        print(prep.protein_coordinates.head())